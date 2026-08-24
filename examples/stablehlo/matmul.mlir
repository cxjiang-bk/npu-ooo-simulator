module {
  func.func @main(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x12xf16>
  ) -> tensor<4x12xf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<4x8xf16>, tensor<8x12xf16>) -> tensor<4x12xf16>
    return %0 : tensor<4x12xf16>
  }
}
