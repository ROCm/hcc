// RUN: %gtest_amp %s -o %t && %t
// Test if there is an accelerator() defined
#include <amp.h>
#include <gtest/gtest.h>

TEST(Accelerator, Creation) {
  Concurrency::accelerator def;
  Concurrency::accelerator_view av = def.create_view();
  // This is our backdoor call in accelerator_view
  // to get the internal OpenCL context out of accelerator_view
  cl_context c = av.clamp_get_context();
  cl_int res;
  cl_uint ret;
  res = clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT,
    sizeof(cl_uint), &ret, NULL);
  // Expect a valid OpenCL context is constructed
  EXPECT_EQ(CL_SUCCESS, res);
  // This is our backdoor call in accelerator_view
  // to get the internal OpenCL queue out of accelerator_view
  cl_command_queue cq = av.clamp_get_command_queue();
  cl_context ret_ctx;
  res = clGetCommandQueueInfo(cq, CL_QUEUE_CONTEXT,
    sizeof(cl_context), &ret_ctx, NULL);
  // Expect res_ctx to be the same
  EXPECT_EQ(ret_ctx, c);

  cl_uint ret1;
  // to get the internal OpenCL context out of accelerator_view
  c = def.get_default_view().clamp_get_context();
  res = clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT,
    sizeof(cl_uint), &ret1, NULL);
  EXPECT_EQ(CL_SUCCESS, res);

  cl_uint ret2;
  // A new view constructed
  Concurrency::accelerator_view av1 = def.get_default_view();
  c = av1.clamp_get_context();
  res = clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT,
    sizeof(cl_uint), &ret2, NULL);
  // Expect a valid OpenCL context is constructed
  EXPECT_EQ(CL_SUCCESS, res);
  EXPECT_EQ(ret1+1, ret2);

  // Test copy constructor
  Concurrency::accelerator_view av2(av1);
  cl_uint ret3;
  res = clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT,
    sizeof(cl_uint), &ret3, NULL);
  // Expect a valid OpenCL context is constructed
  EXPECT_EQ(CL_SUCCESS, res);
  EXPECT_EQ(ret2+1, ret3);
}
