#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
__kernel void computeDFTFloat(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *inreal, __global uchar *inimag, __global uchar *outreal, __global uchar *outimag)
{
  ulong ul_37, ul_38, ul_3, ul_1, ul_2, ul_0, ul_16, ul_14;
  long l_12, l_13, l_35, l_36;
  int i_33, i_34, i_11, i_39, i_10, i_4, i_5, i_6;
  float f_25, f_26, f_27, f_28, f_21, f_22, f_23, f_24, f_29, f_30, f_31, f_32, f_9, f_7, f_8, f_17, f_18, f_19, f_20, f_15;

  // BLOCK 0
  ul_0  =  (ulong) inreal;
  ul_1  =  (ulong) inimag;
  ul_2  =  (ulong) outreal;
  ul_3  =  (ulong) outimag;
  i_4  =  get_global_size(0);
  i_5  =  get_global_id(0);
  
  // BLOCK 1 MERGES [0 5]
  i_6  =  i_5;
  for(;i_6 < 4096;)
  {
    // BLOCK 2
    f_7  =  (float) i_6;
    
    // BLOCK 3 MERGES [2 4]
    f_8  =  0.0F;
    f_9  =  0.0F;
    i_10  =  0;
    for(;i_10 < 4096;)
    {
      // BLOCK 4
      i_11  =  i_10 + 6;
      l_12  =  (long) i_11;
      l_13  =  l_12 << 2;
      ul_14  =  ul_0 + l_13;
      f_15  =  *((__global float *) ul_14);
      ul_16  =  ul_1 + l_13;
      f_17  =  *((__global float *) ul_16);
      f_18  =  *((__global float *) ul_14);
      f_19  =  *((__global float *) ul_16);
      f_20  =  (float) i_10;
      f_21  =  f_20 * 6.2831855F;
      f_22  =  f_21 * f_7;
      f_23  =  f_22 / 4096.0F;
      f_24  =  native_sin(f_23);
      f_25  =  -f_18;
      f_26  =  native_cos(f_23);
      f_27  =  f_26 * f_19;
      f_28  =  fma(f_24, f_25, f_27);
      f_29  =  f_9 + f_28;
      f_30  =  f_17 * f_24;
      f_31  =  fma(f_26, f_15, f_30);
      f_32  =  f_8 + f_31;
      i_33  =  i_10 + 1;
      f_8  =  f_32;
      f_9  =  f_29;
      i_10  =  i_33;
    }  // B4
    
    // BLOCK 5
    i_34  =  i_6 + 6;
    l_35  =  (long) i_34;
    l_36  =  l_35 << 2;
    ul_37  =  ul_2 + l_36;
    *((__global float *) ul_37)  =  f_8;
    ul_38  =  ul_3 + l_36;
    *((__global float *) ul_38)  =  f_9;
    i_39  =  i_4 + i_6;
    i_6  =  i_39;
  }  // B5
  
  // BLOCK 6
  return;
}  // kernel 