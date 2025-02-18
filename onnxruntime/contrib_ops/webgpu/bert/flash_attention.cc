// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status CopyKVCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(total_sequence_length)
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);
  const auto& valid_new_present_shape = shader.AddIndices("valid_new_present_shape");

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.valid_new_present_size")
                            << "  let output_indices = " << valid_new_present_shape.OffsetToIndices("global_idx") << ";\n"
                            << "  let head_size_id = output_indices[3];\n"
                               "  let sequence_id = output_indices[2];\n"
                               "  let num_head_id = output_indices[1];\n"
                               "  let batch = output_indices[0];\n";
  if (has_past_) {
    shader.MainFunctionBody() << "let past_sequence_length = uniforms.past_sequence_length;\n";
    if (past_present_share_buffer_) {
      shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, past_sequence_length + sequence_id, head_size_id)") << ";\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
    } else {
      const auto& past_key = shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
      shader.AddInput("past_value", ShaderUsage::UseUniform);
      shader.MainFunctionBody() << "let present_offset = global_idx;"
                                << "if (sequence_id < past_sequence_length) {\n"
                                << "  let pastOffset = " << past_key.IndicesToOffset("past_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "past_key[pastOffset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "past_value[pastOffset]") << ";\n"
                                << "} else {\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id - past_sequence_length, head_size_id)" : "key_indices_t(batch, sequence_id - past_sequence_length, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n"
                                << "}";
    }
  } else {
    shader.MainFunctionBody() << "  let present_offset = " << (past_present_share_buffer_ ? present_key.IndicesToOffset("output_indices") : "global_idx") << ";\n"
                              << "let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                              << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
  }
  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  bool has_past = (parameters.total_sequence_length_ - parameters.kv_sequence_length_) > 0;
  // parameters.total_sequence_length_ is past_sequence_length + kv_sequence_length.
  // parameters.kv_num_heads_ may be smaller than parameters.num_heads_ when parameters.is_gqa_ is true.
  int num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  // Only copy the new kv data for static kv cache
  int new_sequence_length = has_past && parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  TensorShape valid_new_present_shape{parameters.batch_size_, num_heads, new_sequence_length, parameters.head_size_ / components};
  int64_t valid_new_kv_size = valid_new_present_shape.Size();
  CopyKVCacheProgram program{"CopyKVCache", has_past, parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH, parameters.past_present_share_buffer_};
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_ENFORCE(parameters.qkv_format_ == Q_K_V_BSNH, "qkv format ", parameters.qkv_format_, " is not supported yet in CopyKVCache.");
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, num_heads, parameters.head_size_ / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }
  if (has_past && !parameters.past_present_share_buffer_) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}})
      .AddIndices(valid_new_present_shape);
  program.SetDispatchGroupSize(gsl::narrow<uint32_t>(valid_new_kv_size + 63 / 64))
      .SetWorkgroupSize(64)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_)
      .AddUniformVariables({{static_cast<uint32_t>(valid_new_kv_size)},
                            // Note that when parameters.past_present_share_buffer_ is true, parameters.past_sequence_length_ will become to
                            // max_sequence_length. To get a valid past_sequence_length, we use total_sequence_length - kv_sequence_length.
                            {static_cast<uint32_t>(parameters.total_sequence_length_ - parameters.kv_sequence_length_)}});

  return context.RunProgram(program);
}

Status FlashAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(new_sequence_length)(total_sequence_length)
  //
  //  Expectation is that present_key, and present_value contain past key and values since
  //  we are out of storage buffers a shader can have and both past/present cant be passed.
  // The hidden size of each q head should be a multiple of 4 because shader uses vectorized loads.
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << "const qkv_head_size: u32 = " << qkv_head_size_ << ";\n"
                                    << "const num_heads: u32 =" << qkv_num_heads_ << ";\n";

  shader.AdditionalImplementation() << R"HELPER_FN(
  // For max performance max_k_step should be the same as sg_size, however we might run out of registers
  // for qk_1, qk_2 .. qk_(sg_size). So we cap it at max_k_step (16).
  const max_k_step: u32 = 16u;
  const vec_factor: u32 = 4u;
  const qkv_head_size_vec: u32 = qkv_head_size / vec_factor;
  const min_value : q_element_t = q_element_t(-65504.0);

  // Default SHM usage limit is 16KB in Dawn.
  var<workgroup> k_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 96 * 2 * 16 = 3KB.
  var<workgroup> v_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 96 * 2 * 16 = 3KB.

  // Private memory per lane.
  var<private> q_tile : array<q_value_t, qkv_head_size_vec>;
  var<private> o_tile : array<q_value_t, qkv_head_size_vec>;
  fn loadq(q_idx_global : u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072] the inputs as per onnx MHA
      // This is the layout if TransferBSDToBNSH has not been run.
      let offset = q_idx_global * (qkv_head_size_vec) * num_heads + qkv_head_size_vec * head_idx;
      // Stored as BNSH - which is what webgpu uses after TransferBSDToBNSH has been run.
      //let offset = head_idx * uniforms.new_sequence_length * qkv_head_size_vec + q_idx_global * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx++)
      {
          q_tile[idx] = q[idx+offset];
      }
  }
  fn loadk(k_start : u32, head_idx: u32, local_idx: u32, k_step: u32)
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + k_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec);
          let val = select(q_value_t(0), present_key[offset+idx], k_start + slot < uniforms.total_sequence_length);
          k_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn loadv(v_start : u32, head_idx: u32, local_idx: u32, k_step: u32)
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + v_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec);
          let val  = select(q_value_t(0), present_value[offset+idx], v_start + slot < uniforms.total_sequence_length);
          v_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn writeo(o_idx_global: u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072]
      let offset = o_idx_global * num_heads * qkv_head_size_vec + head_idx * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx ++)
      {
          output[offset+idx] = o_tile[idx];
      }
  }
)HELPER_FN";

  if (has_attention_bias_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
          // Stored as float16[batch_size,num_heads,new_seq_length,total_sequence_length]
          if (q_idx_global >= uniforms.new_sequence_length  || k_idx_global >= uniforms.total_sequence_length) {
              return vec4<q_element_t>(0);
          }
          let offset_base = head_idx * uniforms.new_sequence_length * uniforms.total_sequence_length + q_idx_global * uniforms.total_sequence_length;
          let offset = offset_base + k_idx_global;
          let offset_max = offset_base + uniforms.total_sequence_length;
          let c1 = q_element_t(attention_bias[min(offset, offset_max)]);
          let c2 = q_element_t(attention_bias[min(offset+1, offset_max)]);
          let c3 = q_element_t(attention_bias[min(offset+2, offset_max)]);
          let c4 = q_element_t(attention_bias[min(offset+3, offset_max)]);
          return vec4<q_element_t>(c1,c2,c3,c4);
      }
    )HELPER_FN";
  } else {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
        return vec4<q_element_t>(0);
      }
    )HELPER_FN";
  }

  // Shader is designed to be dispatched as Dispatch(num_heads, new_sequence_length / workgroup_size_x, 1)
  // Each lane/thread is responsible for a single q.
  shader.MainFunctionBody() << R"MAIN_FN(
  let head_idx = workgroup_id.x;
  let capped_sg_id = min(sg_id, max_k_step);
  let capped_sg_size = min(sg_size, max_k_step);

  // Load Q
  let q_idx_global = workgroup_id.y * workgroup_size_x + local_idx;
  let valid_q = q_idx_global < uniforms.new_sequence_length;
  if (valid_q)
  {
    loadq(q_idx_global, head_idx);
  }

  var previous_max : q_element_t = min_value;
  var previous_denom : q_element_t = 0;

  for(var k_start = 0u; k_start < uniforms.total_sequence_length; k_start+=capped_sg_size)
  {
    workgroupBarrier();
    loadk(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    loadv(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    workgroupBarrier();

    // Compute QKt
    var qk_1:vec4<q_element_t>;
    var qk_2:vec4<q_element_t>;
    var qk_3:vec4<q_element_t>;
    var qk_4:vec4<q_element_t>;
    if (sg_size > 8)
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
        qk_3[0] += dot(q_own, subgroupShuffle(k_local, 8));
        qk_3[1] += dot(q_own, subgroupShuffle(k_local, 9));
        qk_3[2] += dot(q_own, subgroupShuffle(k_local, 10));
        qk_3[3] += dot(q_own, subgroupShuffle(k_local, 11));
        qk_4[0] += dot(q_own, subgroupShuffle(k_local, 12));
        qk_4[1] += dot(q_own, subgroupShuffle(k_local, 13));
        qk_4[2] += dot(q_own, subgroupShuffle(k_local, 14));
        qk_4[3] += dot(q_own, subgroupShuffle(k_local, 15));
      }
    }
    else
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
      }
    }

    qk_1 = qk_1 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start, head_idx);
    qk_2 = qk_2 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+4, head_idx);
    if (sg_size > 8)
    {
      qk_3 = qk_3 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+8, head_idx);
      qk_4 = qk_4 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+12, head_idx);
    }

    let seq_causal_length = select(uniforms.total_sequence_length, q_idx_global + 1, uniforms.is_gqa > 0);
    // Neuter qk values where K is out of bounds.
    qk_1[0] = select(min_value, qk_1[0], k_start+0 < seq_causal_length);
    qk_1[1] = select(min_value, qk_1[1], k_start+1 < seq_causal_length);
    qk_1[2] = select(min_value, qk_1[2], k_start+2 < seq_causal_length);
    qk_1[3] = select(min_value, qk_1[3], k_start+3 < seq_causal_length);
    qk_2[0] = select(min_value, qk_2[0], k_start+4 < seq_causal_length);
    qk_2[1] = select(min_value, qk_2[1], k_start+5 < seq_causal_length);
    qk_2[2] = select(min_value, qk_2[2], k_start+6 < seq_causal_length);
    qk_2[3] = select(min_value, qk_2[3], k_start+7 < seq_causal_length);
    if (sg_size > 8)
    {
      qk_3[0] = select(min_value, qk_3[0], k_start+8 < seq_causal_length);
      qk_3[1] = select(min_value, qk_3[1], k_start+9 < seq_causal_length);
      qk_3[2] = select(min_value, qk_3[2], k_start+10 < seq_causal_length);
      qk_3[3] = select(min_value, qk_3[3], k_start+11 < seq_causal_length);
      qk_4[0] = select(min_value, qk_4[0], k_start+12 < seq_causal_length);
      qk_4[1] = select(min_value, qk_4[1], k_start+13 < seq_causal_length);
      qk_4[2] = select(min_value, qk_4[2], k_start+14 < seq_causal_length);
      qk_4[3] = select(min_value, qk_4[3], k_start+15 < seq_causal_length);
    }

    //
    // Compute SoftMax as per Flash Attention technique.
    //
    // Crux of Flash Attention is here, that allows for partial softmax computation,
    // direct update of output and merging with previous results.
    // https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    // Where b is the block size of the tile. Xi is storing QKtranspose for the ith tile.
    // mi_local is the max of Xi. Note: _ in this notation means what follows is a
    // subscript. max_j=1:b (Xi[j]) is the max of Xi[j] for j=1 to b.
    //
    // for i = 1, #tiles do
    //  Xi = Q[k,:] Kt[:, (i-1) b : i b]
    //  mi_local= max_j=1:b (Xi[j])
    //  Mi = max(M_(i-1), mi_local)
    //  d'_i = d'_(i-1) * e^(M_(i-1)-M_i) + Σ_j=1:b e^(Xi[j]-Mi)
    //  o'_i = o'_(i-1) * d'_(i-1) * e^(M_(i-1)-M_i) / d'_i + Σ_j=1:b (e^(Xi[j]-Mi) / d'_i) V[j + (i - 1)b,:]
    // end
    //
    // In the code below:
    // dleft is the first term of d'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i).
    // sum is the second term of the same expression    : Σ_j=1:b e^(Xi[j]-Mi)
    // o_ratio is the part of the first term of o'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i) / d'_i
    //
    var local_max_temp = max(qk_1, qk_2);
    if (sg_size > 8)
    {
      local_max_temp = max(local_max_temp, qk_3);
      local_max_temp = max(local_max_temp, qk_4);
    }
    let local_max = max(max(local_max_temp.x, local_max_temp.y),max(local_max_temp.z, local_max_temp.w));
    let new_max = max(previous_max, local_max);
    qk_1 = q_value_t(exp(vec4<f32>(qk_1) - f32(new_max)));
    qk_2 = q_value_t(exp(vec4<f32>(qk_2) - f32(new_max)));
    if (sg_size > 8) {
      qk_3 = q_value_t(exp(vec4<f32>(qk_3) - f32(new_max)));
      qk_4 = q_value_t(exp(vec4<f32>(qk_4) - f32(new_max)));
    }
    let sum_vec = qk_1 + qk_2 + qk_3 + qk_4;
    let sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

    // Compute lhs term of update di prime and the compute di prime.
    let dleft = previous_denom * exp(previous_max-new_max);
    var d = dleft + sum;
    d = select(d,q_element_t(0.0000001),d==0);
    qk_1 = qk_1 / d;
    qk_2 = qk_2 / d;
    if (sg_size > 8) {
      qk_3 = qk_3 / d;
      qk_4 = qk_4 / d;
    }
    previous_max = new_max;
    previous_denom = d;
    let o_ratio = dleft / d;

    if (sg_size > 8) {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = v_tile[capped_sg_id][i];
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          sum += subgroupShuffle(val, 8) * qk_3[0];
          sum += subgroupShuffle(val, 9) * qk_3[1];
          sum += subgroupShuffle(val, 10) * qk_3[2];
          sum += subgroupShuffle(val, 11) * qk_3[3];
          sum += subgroupShuffle(val, 12) * qk_4[0];
          sum += subgroupShuffle(val, 13) * qk_4[1];
          sum += subgroupShuffle(val, 14) * qk_4[2];
          sum += subgroupShuffle(val, 15) * qk_4[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
    else
    {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = select(vec4<q_element_t>(0), v_tile[capped_sg_id][i], k_start + capped_sg_id < seq_causal_length);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
  }

  if (valid_q) {
    writeo(q_idx_global, head_idx);
  }
)MAIN_FN";

  return Status::OK();
}

Status AttentionQKTProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << "const Tile_Size = " << tile_size_ << ";\n";
  shader.AdditionalImplementation() << "const Tile_N_Size = " << tile_size_ * tile_size_ << ";\n";
  shader.AdditionalImplementation() << R"ADDNL_FN(
var<workgroup> tileQ: array<q_value_t, Tile_Size>;
var<workgroup> inner_qk_values: array<array<q_element_t, Tile_Size>, Tile_N_Size>;
var<workgroup> tile_qk: array<q_element_t, Tile_N_Size>;
)ADDNL_FN";

  if (has_attention_bias_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(idx: u32) -> q_element_t
      {
        return attention_bias[idx];
      }
    )HELPER_FN";
  } else {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(idx: u32) -> q_element_t
      {
        return q_element_t(0);
      }
    )HELPER_FN";
  }

  shader.MainFunctionBody() << R"MAIN_FN(
    let total_seq_offset = workgroup_id.x * Tile_N_Size;
    let qOffset = workgroup_id.z * uniforms.head_size_vec;
    var total_sequence_length = uniforms.total_sequence_length;
    let presentKeyOffset = (workgroup_id.z / uniforms.n_reps) * uniforms.present_sequence_length * uniforms.head_size_vec;
    for (var w: u32 = 0u; w < uniforms.head_size_vec; w += Tile_Size) {
      if (local_idx < Tile_Size && w + local_idx < uniforms.head_size_vec) {
        tileQ[local_idx] = q[qOffset + w + local_idx];
      }
      workgroupBarrier();
      for (var row_offset = 0u; row_offset < Tile_N_Size; row_offset += Tile_Size) {
        if (total_seq_offset + row_offset + local_id.y < total_sequence_length && w + local_id.x < uniforms.head_size_vec) {
          inner_qk_values[row_offset + local_id.y][local_id.x] += dot(present_key[presentKeyOffset + (total_seq_offset + row_offset + local_id.y) * uniforms.head_size_vec + w + local_id.x], tileQ[local_id.x]);
        }
      }
      workgroupBarrier();
    }
    var sum = q_element_t(0);
    for (var i = 0u; i < Tile_Size; i++) {
      sum += inner_qk_values[local_idx][i];
    }
    if ((total_seq_offset + local_idx) < total_sequence_length) {
      let output_idx = workgroup_id.z * total_sequence_length + total_seq_offset + local_idx;
      sum = sum * q_element_t(uniforms.alpha) + loadAttentionBias(output_idx);
      tile_qk[local_idx] = sum;
      output[output_idx] = sum;
    }

    workgroupBarrier();
    var l_max = f32(-3.402823e+38f);
    var l_sum = f32(0);
    for (var i = 0u; i < Tile_N_Size && (total_seq_offset + i) < total_sequence_length; i++) {
      l_max = max(l_max, f32(tile_qk[i]));
    }
    for (var i = 0u; i < Tile_N_Size && (total_seq_offset + i) < total_sequence_length; i++) {
      l_sum += exp(f32(tile_qk[i]) - l_max);
    }
    if (local_idx == 0u) {
      let meta_offset = workgroup_id.z * uniforms.splited_k + workgroup_id.x;
      metadata[meta_offset] = metadata_value_t(metadata_element_t(l_max), metadata_element_t(l_sum));
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeAttentionQKT(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                           const Tensor* attention_bias, Tensor* output, Tensor* present_key, Tensor* metadata,
                           const WebgpuAttentionParameters& parameters, int splited_k) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  constexpr int tile_size = 8;
  constexpr int tile_n_size = 64;
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);

  AttentionQKTProgram program{"AttentionQKT", has_attention_bias, tile_size};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::Rank},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = (parameters.head_size_ + components - 1) / components;
  program.SetDispatchGroupSize((parameters.total_sequence_length_ + tile_n_size - 1) / tile_n_size,
                               parameters.sequence_length_,
                               parameters.batch_size_ * parameters.num_heads_)
      .SetWorkgroupSize(tile_size, tile_size)
      .CacheHint(std::to_string(tile_size), has_attention_bias, components)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<float>(alpha)},
                            {static_cast<uint32_t>(parameters.is_gqa_ ? parameters.seqlen_present_kv_cache_ : parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {static_cast<uint32_t>(splited_k)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeSplitKProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("metadata", ShaderUsage::UseUniform);
  shader.AddInput("qk", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("out_split_k", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << "const Head_Size_Vec = " << head_size_vec_ << ";\n";
  shader.AdditionalImplementation() << "const Tile_Size = " << tile_size_ << ";\n";
  shader.AdditionalImplementation() << "const Tile_N_Size = " << tile_size_ * tile_size_ << ";\n";
  shader.AdditionalImplementation() << R"HELPER_FN(
var<workgroup> tile_qk: array<qk_value_t, Tile_N_Size>;
var<workgroup> tile_output: array<present_value_value_t, Head_Size_Vec>;
var<workgroup> qkv_values: array<array<present_value_value_t, Tile_Size>, Tile_Size>;

  )HELPER_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
    let local_row = local_idx / Tile_Size;
    let local_col = local_idx % Tile_Size;
    let total_seq_offset = workgroup_id.x * Tile_N_Size;
    var total_sequence_length = uniforms.total_sequence_length;
    let presentValueOffset = (workgroup_id.z / uniforms.n_reps) * uniforms.head_size_vec * uniforms.present_sequence_length;

    var g_max = f32(-3.402823e+38f);
    for (var i = 0u; i < uniforms.splited_k; i++)
    {
      let meta_offset = workgroup_id.z * uniforms.splited_k + i;
      g_max = max(g_max, f32(metadata[meta_offset].x));
    }
    var g_sum = f32(0);
    for (var i = 0u; i < uniforms.splited_k; i++)
    {
      let meta_offset = workgroup_id.z * uniforms.splited_k + i;
      let m_value = metadata[meta_offset];
      g_sum += exp(f32(m_value.x) - g_max) * f32(m_value.y);
    }

    var value = present_value_value_t(0);

    if (total_seq_offset + local_idx < total_sequence_length) {
       tile_qk[local_idx] = qk_value_t(exp(f32(qk[workgroup_id.z * total_sequence_length + total_seq_offset + local_idx]) - g_max) / g_sum);
    }
    workgroupBarrier();

    for (var w: u32 = 0u; w < uniforms.head_size_vec; w += Tile_Size) {
      value = present_value_value_t(0);
      qkv_values[local_row][local_col] = value;
      workgroupBarrier();

      for (var row_offset = 0u; row_offset < Tile_N_Size; row_offset += Tile_Size) {
        if (total_seq_offset + row_offset + local_row < total_sequence_length && w + local_col < uniforms.head_size_vec) {
          value += present_value[presentValueOffset + (total_seq_offset + row_offset + local_row) * uniforms.head_size_vec + w + local_col] * tile_qk[row_offset + local_row];
        }
      }

      qkv_values[local_row][local_col] = value;
      workgroupBarrier();

      if (local_idx < Tile_Size) {
        for (var i = 0u; i < Tile_Size; i++) {
          tile_output[w + local_idx] += qkv_values[i][local_idx];
        }
      }
      workgroupBarrier();
    }

    for (var i = local_idx; i < uniforms.head_size_vec; i += Tile_N_Size) {
      // workgroup_id.x is the current split_k index
      let out_offset = workgroup_id.z * uniforms.splited_k * uniforms.head_size_vec + workgroup_id.x * uniforms.head_size_vec + i;
      out_split_k[out_offset] = tile_output[i];
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeFlashAttentionDecodeSplitK(onnxruntime::webgpu::ComputeContext& context,
                                         const Tensor* metadata,
                                         const Tensor* qk,
                                         Tensor* out_split_k,
                                         Tensor* present_value,
                                         const WebgpuAttentionParameters& parameters,
                                         int split_k) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_n_size = tile_size * tile_size;
  int head_size_vec = parameters.v_head_size_ / components;
  FlashAttentionDecodeSplitKProgram program{"FlashAttentionDecodeSplitK", tile_size, head_size_vec};
  program.AddInputs({{metadata, ProgramTensorMetadataDependency::TypeAndRank, 2},
                     {qk, ProgramTensorMetadataDependency::TypeAndRank},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  program.AddOutputs({{out_split_k, ProgramTensorMetadataDependency::TypeAndRank, components}});  // [B, N, split_k, head_size]

  program.SetDispatchGroupSize((parameters.total_sequence_length_ + tile_n_size - 1) / tile_n_size,
                               parameters.sequence_length_,
                               parameters.batch_size_ * parameters.num_heads_)
      .CacheHint(std::to_string(tile_size))
      .SetWorkgroupSize(tile_n_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(head_size_vec)},
                            {static_cast<uint32_t>(parameters.is_gqa_ ? parameters.seqlen_present_kv_cache_ : parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {static_cast<uint32_t>(split_k)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.AdditionalImplementation() << "const TILE_SIZE = " << tile_size_ << ";\n";
  shader.AdditionalImplementation() << R"HELPER_FN(
var<workgroup> tile_input: array<array<output_value_t, TILE_SIZE>, TILE_SIZE>;
  )HELPER_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
    let head_size_offset = workgroup_id.x * TILE_SIZE;
    let in_offset = workgroup_id.z * uniforms.splited_k * uniforms.head_size_vec;
    var value = output_value_t(0);
    let local_row = local_idx / TILE_SIZE;
    let local_col = local_idx % TILE_SIZE;

    for (var r = 0u; r < uniforms.splited_k; r += TILE_SIZE) {
      if (r + local_row < uniforms.splited_k && head_size_offset + local_col < uniforms.head_size_vec) {
        value += input[in_offset + (r + local_row) * uniforms.head_size_vec + head_size_offset + local_col];
      }
    }

    tile_input[local_row][local_col] = value;
    workgroupBarrier();

    if (local_idx < TILE_SIZE && head_size_offset + local_idx < uniforms.head_size_vec) {
      value = output_value_t(0);
      for (var i = 0u; i < TILE_SIZE; i++) {
        value += tile_input[i][local_idx];
      }
      let output_id = workgroup_id.z * uniforms.head_size_vec + head_size_offset + local_idx;
      output[output_id] = value;
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeFlashAttentionDecodeReduce(onnxruntime::webgpu::ComputeContext& context,
                                         const Tensor* out_split_k,
                                         Tensor* output,
                                         const WebgpuAttentionParameters& parameters,
                                         int split_k) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_n_size = tile_size * components;
  FlashAttentionDecodeReduceProgram program{"FlashAttentionDecodeReduce", tile_size};
  program.AddInputs({{out_split_k, ProgramTensorMetadataDependency::TypeAndRank, components}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});

  program.SetDispatchGroupSize((parameters.v_head_size_ + tile_n_size - 1) / tile_n_size,
                               parameters.sequence_length_,
                               parameters.batch_size_ * parameters.num_heads_)
      .CacheHint(std::to_string(tile_size))
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            {static_cast<uint32_t>(split_k)}});

  return context.RunProgram(program);
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value));

  if (parameters.sequence_length_ > 1) {
    const uint32_t tile_size = 64;
    bool has_attention_bias = attention_bias != nullptr;
    FlashAttentionProgram program{"FlashAttention", has_attention_bias, parameters.head_size_, parameters.num_heads_};
    program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    if (has_attention_bias) {
      program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
    }
    program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                  : parameters.scale_;
    std::string cache_hint = std::to_string(has_attention_bias) +
                             std::to_string(parameters.head_size_) +
                             std::to_string(parameters.num_heads_);
    program.SetDispatchGroupSize(parameters.num_heads_, (parameters.sequence_length_ + tile_size - 1) / tile_size, 1)
        .SetWorkgroupSize(tile_size)
        .CacheHint(cache_hint)
        .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(parameters.past_present_share_buffer_ ? parameters.past_sequence_length_ : parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(parameters.is_gqa_ ? 1 : 0)},
                              {static_cast<uint32_t>(parameters.n_reps)},
                              {alpha}});

    return context.RunProgram(program);
  }

  const TensorShapeVector qk_dims({parameters.batch_size_, parameters.num_heads_,
                                   parameters.sequence_length_, parameters.total_sequence_length_});
  const TensorShape qk_shape(qk_dims);
  Tensor qk = context.CreateGPUTensor(Q->DataType(), qk_shape);
  constexpr int tile_size = 64;
  const int splited_k = (parameters.total_sequence_length_ + tile_size - 1) / tile_size;
  const TensorShapeVector metadata_dims({parameters.batch_size_, parameters.num_heads_,
                                         splited_k, 2});
  const TensorShape metadata_shape(metadata_dims);
  Tensor metadata = context.CreateGPUTensor(Q->DataType(), metadata_shape);
  ORT_RETURN_IF_ERROR(ComputeAttentionQKT(context, Q, attention_bias, &qk, present_key, &metadata,
                                          parameters, splited_k));

  const TensorShapeVector out_split_k_dims({parameters.batch_size_, parameters.num_heads_, splited_k, parameters.head_size_});
  const TensorShape out_split_k_shape(out_split_k_dims);
  Tensor out_split_k = context.CreateGPUTensor(Q->DataType(), out_split_k_shape);
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeSplitK(context, &metadata, &qk, &out_split_k, present_value, parameters, splited_k));
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeReduce(context, &out_split_k, output, parameters, splited_k));

  return Status::OK();
}

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return parameters.batch_size_ == 1 &&
         !parameters.is_packed_qkv_ &&
         parameters.head_size_ == parameters.v_head_size_ &&
         // Only use flash decoding when total_sequence_length is large. Below threshold is empirical value.
         // [total_sequence_length / 64] and [head_size / 16] are the dispatched workgroups on dimension X with and without flash attention.
         // And when the former is 2x larger than latter, switching to flash attention can improve the parallelism by dispatching more workgroups.
         (parameters.sequence_length_ > 1 || parameters.total_sequence_length_ / 64 >= parameters.head_size_ / 16) &&
         bias == nullptr &&
         context.Device().HasFeature(wgpu::FeatureName::Subgroups) &&
         present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
         present_value->SizeInBytes() > 0 && parameters.head_size_ % 4 == 0;
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
