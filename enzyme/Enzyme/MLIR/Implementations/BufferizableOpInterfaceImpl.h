//===- BufferizableOpInterfaceImpl.h - Impl registrations -* C++ -*--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of
// BufferizableOpInterface for enzyme dialect ops
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ENZYME_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_ENZYME_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace enzyme {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir

#endif // MLIR_DIALECT_ENZYME_BUFFERIZABLEOPINTERFACEIMPL_H
