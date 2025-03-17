// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { Attribute } from '../../onnxruntime/fbs/attribute.js';
import { NodeType } from '../../onnxruntime/fbs/node-type.js';

export class Node {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): Node {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsNode(bb: flatbuffers.ByteBuffer, obj?: Node): Node {
    return (obj || new Node()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsNode(bb: flatbuffers.ByteBuffer, obj?: Node): Node {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new Node()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  name(): string | null;
  name(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  name(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  docString(): string | null;
  docString(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  docString(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  domain(): string | null;
  domain(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  domain(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 8);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  sinceVersion(): number {
    const offset = this.bb!.__offset(this.bb_pos, 10);
    return offset ? this.bb!.readInt32(this.bb_pos + offset) : 0;
  }

  index(): number {
    const offset = this.bb!.__offset(this.bb_pos, 12);
    return offset ? this.bb!.readUint32(this.bb_pos + offset) : 0;
  }

  opType(): string | null;
  opType(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  opType(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 14);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  type(): NodeType {
    const offset = this.bb!.__offset(this.bb_pos, 16);
    return offset ? this.bb!.readInt32(this.bb_pos + offset) : NodeType.Primitive;
  }

  executionProviderType(): string | null;
  executionProviderType(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  executionProviderType(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 18);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  inputs(index: number): string;
  inputs(index: number, optionalEncoding: flatbuffers.Encoding): string | Uint8Array;
  inputs(index: number, optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 20);
    return offset ? this.bb!.__string(this.bb!.__vector(this.bb_pos + offset) + index * 4, optionalEncoding) : null;
  }

  inputsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 20);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  outputs(index: number): string;
  outputs(index: number, optionalEncoding: flatbuffers.Encoding): string | Uint8Array;
  outputs(index: number, optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 22);
    return offset ? this.bb!.__string(this.bb!.__vector(this.bb_pos + offset) + index * 4, optionalEncoding) : null;
  }

  outputsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 22);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  attributes(index: number, obj?: Attribute): Attribute | null {
    const offset = this.bb!.__offset(this.bb_pos, 24);
    return offset
      ? (obj || new Attribute()).__init(
          this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4),
          this.bb!,
        )
      : null;
  }

  attributesLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 24);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  inputArgCounts(index: number): number | null {
    const offset = this.bb!.__offset(this.bb_pos, 26);
    return offset ? this.bb!.readInt32(this.bb!.__vector(this.bb_pos + offset) + index * 4) : 0;
  }

  inputArgCountsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 26);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  inputArgCountsArray(): Int32Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 26);
    return offset
      ? new Int32Array(
          this.bb!.bytes().buffer,
          this.bb!.bytes().byteOffset + this.bb!.__vector(this.bb_pos + offset),
          this.bb!.__vector_len(this.bb_pos + offset),
        )
      : null;
  }

  implicitInputs(index: number): string;
  implicitInputs(index: number, optionalEncoding: flatbuffers.Encoding): string | Uint8Array;
  implicitInputs(index: number, optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 28);
    return offset ? this.bb!.__string(this.bb!.__vector(this.bb_pos + offset) + index * 4, optionalEncoding) : null;
  }

  implicitInputsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 28);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  static startNode(builder: flatbuffers.Builder) {
    builder.startObject(13);
  }

  static addName(builder: flatbuffers.Builder, nameOffset: flatbuffers.Offset) {
    builder.addFieldOffset(0, nameOffset, 0);
  }

  static addDocString(builder: flatbuffers.Builder, docStringOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, docStringOffset, 0);
  }

  static addDomain(builder: flatbuffers.Builder, domainOffset: flatbuffers.Offset) {
    builder.addFieldOffset(2, domainOffset, 0);
  }

  static addSinceVersion(builder: flatbuffers.Builder, sinceVersion: number) {
    builder.addFieldInt32(3, sinceVersion, 0);
  }

  static addIndex(builder: flatbuffers.Builder, index: number) {
    builder.addFieldInt32(4, index, 0);
  }

  static addOpType(builder: flatbuffers.Builder, opTypeOffset: flatbuffers.Offset) {
    builder.addFieldOffset(5, opTypeOffset, 0);
  }

  static addType(builder: flatbuffers.Builder, type: NodeType) {
    builder.addFieldInt32(6, type, NodeType.Primitive);
  }

  static addExecutionProviderType(builder: flatbuffers.Builder, executionProviderTypeOffset: flatbuffers.Offset) {
    builder.addFieldOffset(7, executionProviderTypeOffset, 0);
  }

  static addInputs(builder: flatbuffers.Builder, inputsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(8, inputsOffset, 0);
  }

  static createInputsVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startInputsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static addOutputs(builder: flatbuffers.Builder, outputsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(9, outputsOffset, 0);
  }

  static createOutputsVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startOutputsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static addAttributes(builder: flatbuffers.Builder, attributesOffset: flatbuffers.Offset) {
    builder.addFieldOffset(10, attributesOffset, 0);
  }

  static createAttributesVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startAttributesVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static addInputArgCounts(builder: flatbuffers.Builder, inputArgCountsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(11, inputArgCountsOffset, 0);
  }

  static createInputArgCountsVector(builder: flatbuffers.Builder, data: number[] | Int32Array): flatbuffers.Offset;
  /**
   * @deprecated This Uint8Array overload will be removed in the future.
   */
  static createInputArgCountsVector(builder: flatbuffers.Builder, data: number[] | Uint8Array): flatbuffers.Offset;
  static createInputArgCountsVector(
    builder: flatbuffers.Builder,
    data: number[] | Int32Array | Uint8Array,
  ): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addInt32(data[i]!);
    }
    return builder.endVector();
  }

  static startInputArgCountsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static addImplicitInputs(builder: flatbuffers.Builder, implicitInputsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(12, implicitInputsOffset, 0);
  }

  static createImplicitInputsVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startImplicitInputsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static endNode(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }

  static createNode(
    builder: flatbuffers.Builder,
    nameOffset: flatbuffers.Offset,
    docStringOffset: flatbuffers.Offset,
    domainOffset: flatbuffers.Offset,
    sinceVersion: number,
    index: number,
    opTypeOffset: flatbuffers.Offset,
    type: NodeType,
    executionProviderTypeOffset: flatbuffers.Offset,
    inputsOffset: flatbuffers.Offset,
    outputsOffset: flatbuffers.Offset,
    attributesOffset: flatbuffers.Offset,
    inputArgCountsOffset: flatbuffers.Offset,
    implicitInputsOffset: flatbuffers.Offset,
  ): flatbuffers.Offset {
    Node.startNode(builder);
    Node.addName(builder, nameOffset);
    Node.addDocString(builder, docStringOffset);
    Node.addDomain(builder, domainOffset);
    Node.addSinceVersion(builder, sinceVersion);
    Node.addIndex(builder, index);
    Node.addOpType(builder, opTypeOffset);
    Node.addType(builder, type);
    Node.addExecutionProviderType(builder, executionProviderTypeOffset);
    Node.addInputs(builder, inputsOffset);
    Node.addOutputs(builder, outputsOffset);
    Node.addAttributes(builder, attributesOffset);
    Node.addInputArgCounts(builder, inputArgCountsOffset);
    Node.addImplicitInputs(builder, implicitInputsOffset);
    return Node.endNode(builder);
  }
}
