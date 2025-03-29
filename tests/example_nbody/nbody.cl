#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#pragma OPENCL EXTENSION cl_khr_fp16 : enable  
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable  
__kernel void nBody(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __private int numBodies, __global uchar *refPos, __global uchar *refVel)
{
  float f_100, f_96, f_97, f_98, f_30, f_31, f_24, f_52, f_53, f_54, f_55, f_51, f_60, f_61, f_62, f_63, f_56, f_59, f_37, f_38, f_39, f_32, f_44, f_45, f_46, f_84, f_85, f_86, f_87, f_80, f_81, f_82, f_83, f_92, f_93, f_94, f_95, f_89, f_90, f_91, f_69, f_70, f_71, f_64, f_65, f_66, f_67, f_76, f_78, f_79, f_72, f_73, f_74, f_75; 
  double d_57, d_58; 
  long l_41, l_42, l_35, l_34, l_27, l_28, l_20, l_21, l_16, l_48, l_17, l_49, l_12, l_13; 
  int i_68, i_101, i_33, i_26, i_25, i_23, i_19, i_15, i_47, i_10, i_11, i_8, i_40, i_9, i_6, i_7; 
  ulong ul_88, ul_14, ul_29, ul_77, ul_43, ul_18, ul_50, ul_1, ul_0, ul_22, ul_36, ul_99; 

  // BLOCK 0
  ul_0  =  (ulong) refPos;
  ul_1  =  (ulong) refVel;
  __private float ul_2[3];
  __private float* ul_3 = ul_2;
  __private float ul_4[3];
  __private float* ul_5 = ul_4;
  i_6  =  get_global_size(0);
  i_7  =  get_global_id(0);
  i_8  =  _kernel_context[0];
  // BLOCK 1 MERGES [0 5 ]
  i_9  =  i_7;
  for(;i_9 < i_8;)
  {
    // BLOCK 2
    ul_3[0]  =  0.0F;
    ul_3[1]  =  0.0F;
    ul_3[2]  =  0.0F;
    i_10  =  i_9 << 2;
    i_11  =  i_10 + 8;
    l_12  =  (long) i_11;
    l_13  =  l_12 << 2;
    ul_14  =  ul_0 + l_13;
    i_15  =  i_10 + 7;
    l_16  =  (long) i_15;
    l_17  =  l_16 << 2;
    ul_18  =  ul_0 + l_17;
    i_19  =  i_10 + 6;
    l_20  =  (long) i_19;
    l_21  =  l_20 << 2;
    ul_22  =  ul_0 + l_21;
    // BLOCK 3 MERGES [2 4 ]
    i_23  =  0;
    f_24  =  0.0F;
    for(;i_23 < i_8;)
    {
      // BLOCK 4
      i_25  =  i_23 << 2;
      i_26  =  i_25 + 6;
      l_27  =  (long) i_26;
      l_28  =  l_27 << 2;
      ul_29  =  ul_0 + l_28;
      f_30  =  *((__global float *) ul_29);
      f_31  =  *((__global float *) ul_22);
      f_32  =  f_30 - f_31;
      ul_5[0]  =  f_32;
      i_33  =  i_25 + 7;
      l_34  =  (long) i_33;
      l_35  =  l_34 << 2;
      ul_36  =  ul_0 + l_35;
      f_37  =  *((__global float *) ul_36);
      f_38  =  *((__global float *) ul_18);
      f_39  =  f_37 - f_38;
      ul_5[1]  =  f_39;
      i_40  =  i_25 + 8;
      l_41  =  (long) i_40;
      l_42  =  l_41 << 2;
      ul_43  =  ul_0 + l_42;
      f_44  =  *((__global float *) ul_43);
      f_45  =  *((__global float *) ul_14);
      f_46  =  f_44 - f_45;
      ul_5[2]  =  f_46;
      i_47  =  i_25 + 9;
      l_48  =  (long) i_47;
      l_49  =  l_48 << 2;
      ul_50  =  ul_0 + l_49;
      f_51  =  *((__global float *) ul_50);
      f_52  =  ul_3[0];
      f_53  =  fma(f_32, f_32, 0.0F);
      f_54  =  fma(f_39, f_39, f_53);
      f_55  =  fma(f_46, f_46, f_54);
      f_56  =  f_55 + 500.0F;
      d_57  =  (double) f_56;
      d_58  =  rsqrt(d_57);
      f_59  =  (float) d_58;
      f_60  =  f_59 * f_59;
      f_61  =  f_60 * f_59;
      f_62  =  f_61 * f_51;
      f_63  =  fma(f_62, f_32, f_52);
      ul_3[0]  =  f_63;
      f_64  =  ul_3[1];
      f_65  =  fma(f_62, f_39, f_64);
      ul_3[1]  =  f_65;
      f_66  =  ul_3[2];
      f_67  =  fma(f_62, f_46, f_66);
      ul_3[2]  =  f_67;
      i_68  =  i_23 + 1;
      i_23  =  i_68;
      f_24  =  f_63;
    }  // B4
    // BLOCK 5
    f_69  =  *((__global float *) ul_22);
    f_70  =  *((__global float *) ul_22);
    f_71  =  f_24 * 0.5F;
    f_72  =  f_71 * 0.005F;
    f_73  =  fma(f_70, 0.005F, f_69);
    f_74  =  fma(f_72, 0.005F, f_73);
    *((__global float *) ul_22)  =  f_74;
    f_75  =  *((__global float *) ul_22);
    f_76  =  ul_3[0];
    ul_77  =  ul_1 + l_21;
    f_78  =  fma(f_76, 0.005F, f_75);
    *((__global float *) ul_77)  =  f_78;
    f_79  =  *((__global float *) ul_18);
    f_80  =  *((__global float *) ul_18);
    f_81  =  ul_3[1];
    f_82  =  f_81 * 0.5F;
    f_83  =  f_82 * 0.005F;
    f_84  =  fma(f_80, 0.005F, f_79);
    f_85  =  fma(f_83, 0.005F, f_84);
    *((__global float *) ul_18)  =  f_85;
    f_86  =  *((__global float *) ul_18);
    f_87  =  ul_3[1];
    ul_88  =  ul_1 + l_17;
    f_89  =  fma(f_87, 0.005F, f_86);
    *((__global float *) ul_88)  =  f_89;
    f_90  =  *((__global float *) ul_14);
    f_91  =  *((__global float *) ul_14);
    f_92  =  ul_3[2];
    f_93  =  f_92 * 0.5F;
    f_94  =  f_93 * 0.005F;
    f_95  =  fma(f_91, 0.005F, f_90);
    f_96  =  fma(f_94, 0.005F, f_95);
    *((__global float *) ul_14)  =  f_96;
    f_97  =  *((__global float *) ul_14);
    f_98  =  ul_3[2];
    ul_99  =  ul_1 + l_13;
    f_100  =  fma(f_98, 0.005F, f_97);
    *((__global float *) ul_99)  =  f_100;
    i_101  =  i_6 + i_9;
    i_9  =  i_101;
  }  // B5
  // BLOCK 6
  return;
}  //  kernel

