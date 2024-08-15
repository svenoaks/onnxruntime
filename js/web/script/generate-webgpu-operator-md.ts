// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import fs from 'fs';
import { EOL } from 'os';
import path from 'path';

// The following variable allows to insert comments per operator

const COMMENTS: Record<string, string> = {
  AveragePool: 'need perf optimization; need implementing activation',
  MaxPool: 'need perf optimization; need implementing activation',
  Conv: 'need perf optimization; conv3d is not supported; need implementing activation',
  ConvTranspose: 'need perf optimization; ConvTranspose3d is not supported; need implementing activation',
  Transpose: 'need perf optimization',
  Reshape: 'no GPU kernel',
  Shape: 'no GPU kernel; an ORT warning is generated - need to fix',
  Resize: 'CoordinateTransformMode align_corners is not supported with downsampling',
  Attention: 'need implementing mask and past/present',
  MultiHeadAttention: 'need implementing mask and past/present',
};

/* eslint-disable max-len */
const MATCHERS = [
  /class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME\(\s*(?<ep>\w+),\s*(?<opsetDomain>\w+),\s*(?<opsetVersionStart>\d+),\s*(?<opsetVersionEnd>\d+),\s*(?<op>\w+)\)/g,
  /class ONNX_OPERATOR_KERNEL_CLASS_NAME\(\s*(?<ep>\w+),\s*(?<opsetDomain>\w+),\s*(?<opsetVersion>\d+),\s*(?<op>\w+)\)/g,
  /class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME\(\s*(?<ep>\w+),\s*(?<opsetDomain>\w+),\s*(?<opsetVersionStart>\d+),\s*(?<opsetVersionEnd>\d+),\s*(?<type>\w+),\s*(?<op>\w+)\)/g,
  /class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME\(\s*(?<ep>\w+),\s*(?<opsetDomain>\w+),\s*(?<opsetVersion>\d+),\s*(?<type>\w+),\s*(?<op>\w+)\)/g,
  /class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME\(\s*(?<ep>\w+),\s*(?<opsetDomain>\w+),\s*(?<opsetVersion>\d+),\s*(?<type1>\w+),\s*(?<type2>\w+),\s*(?<op>\w+)\)/g,
];
/* eslint-enable max-len */

const ALL_REGISTERED_OPERATORS: Map<
  string,
  {
    opset: Map<string, Array<[number, number | undefined]>>;
    comments: string;
  }
> = new Map();

// parse js_execution_provider.cc
const JS_EXECUTION_PROVIDER_CONTENTS =
  fs.readFileSync(path.join(__dirname, '../../../onnxruntime/core/providers/js/js_execution_provider.cc'), 'utf8') +
  fs.readFileSync(path.join(__dirname, '../../../onnxruntime/contrib_ops/js/js_contrib_kernels.cc'), 'utf8');
MATCHERS.forEach((m) => {
  for (const match of JS_EXECUTION_PROVIDER_CONTENTS.matchAll(m)) {
    const groups = match.groups!;
    const { ep, opsetDomain, opsetVersion, opsetVersionStart, opsetVersionEnd, op } = groups;

    if (ep !== 'kJsExecutionProvider') {
      throw new Error(`invalid EP registration for EP name: ${ep}`);
    }
    let domain = '';
    switch (opsetDomain) {
      case 'kOnnxDomain':
        domain = 'ai.onnx';
        break;
      case 'kMSInternalNHWCDomain':
        domain = 'com.ms.internal.nhwc';
        break;
      case 'kMSDomain':
        domain = 'com.microsoft';
        break;
      default:
        throw new Error(`not supported domain: ${opsetDomain}`);
    }

    let opInfo = ALL_REGISTERED_OPERATORS.get(op);
    if (!opInfo) {
      opInfo = { opset: new Map(), comments: COMMENTS[op] };
      ALL_REGISTERED_OPERATORS.set(op, opInfo);
    }
    const { opset } = opInfo;
    let currentDomainInfo = opset.get(domain);
    if (!currentDomainInfo) {
      currentDomainInfo = [];
      opset.set(domain, currentDomainInfo);
    }
    if (opsetVersion) {
      currentDomainInfo.push([parseInt(opsetVersion, 10), undefined]);
    } else {
      currentDomainInfo.push([parseInt(opsetVersionStart, 10), parseInt(opsetVersionEnd, 10)]);
    }
    currentDomainInfo.sort((a, b) => a[0] - b[0]);
  }
});

const doc = fs.createWriteStream(path.join(__dirname, '../docs/webgpu-operators.md'));
doc.write(`## Operators Support Table${EOL}${EOL}`);
doc.write(`The following table shows ONNX
operators and the supported opset domain/versions in WebGPU EP by ONNX Runtime Web. For example,
\`4-6, 8+\` means ONNX Runtime Web currently support opset version 4 to 6, 8 and above.${EOL}${EOL}`);
doc.write(`*This file is automatically generated from the
def files via [this script](../script/generate-webgpu-operator-md.ts).
Do not modify directly.*${EOL}${EOL}`);
doc.write(`| Operator | Opset | Comments |${EOL}`);
doc.write(`|:--------:|:-------------:|-----|${EOL}`);

Array.from(ALL_REGISTERED_OPERATORS.keys())
  .sort()
  .forEach((op) => {
    const { opset, comments } = ALL_REGISTERED_OPERATORS.get(op)!;
    const opsetString = Array.from(opset.keys())
      .sort()
      .map(
        (domain) =>
          `${domain}(${[
            ...new Set(
              opset
                .get(domain)!
                .map((ver) => (ver[1] ? (ver[0] === ver[1] ? `${ver[0]}` : `${ver[0]}-${ver[1]}`) : `${ver[0]}+`)),
            ),
          ].join(',')})`,
      )
      .join('; ');
    doc.write(`| ${op} | ${opsetString} | ${comments ?? ''} |${EOL}`);
  });
doc.end();
