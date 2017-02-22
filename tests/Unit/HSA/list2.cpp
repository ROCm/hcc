
// RUN: %hc %s -o %t.out && %t.out

#include <vector>
#include <iostream>
#include <amp.h>
#include <malloc.h>
#include <string.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

typedef signed short ee_s16;
typedef unsigned short ee_u16;
typedef signed int ee_s32;
typedef double ee_f32;
typedef unsigned char ee_u8;
typedef unsigned int ee_u32;
typedef unsigned long long ee_ptr_int;
typedef size_t ee_size_t;
/* align an offset to point to a 32b value */
#define align_mem(x) (void *)(4 + (((ee_ptr_int)(x) - 1) & ~3))


#if 0
/* list data structures */
typedef struct list_data_s {
	ee_s16 data16;
	ee_s16 idx;
} list_data;

typedef struct list_head_s {
	struct list_head_s *next;
	struct list_data_s *info;
} list_head;
#endif

class list_data {
public:
	ee_s16 data16;
	ee_s16 idx;
};

class list_head {
public:
	list_head *next;
	list_data *info;
};




#define NUM_LIST_NODES (20)


list_head *list_insert_new(list_head *llist_head, list_head *newitem, list_data *info, int idx) restrict (amp, cpu) {

	newitem->next=llist_head[idx].next;
	llist_head[idx].next=newitem;
	newitem->info=info;

	return llist_head;
}

bool test() {
  list_head *llist = (list_head *) malloc(NUM_LIST_NODES * sizeof(list_head));
  list_data *ldata = (list_data *) malloc(NUM_LIST_NODES * sizeof(list_data));
  memset(ldata, 0, NUM_LIST_NODES * sizeof(list_data));
  for (int i = 0; i < NUM_LIST_NODES; i++) {
	  ldata[i].data16 = i;
      llist[i].info = &ldata[i];
      llist[i].next = &llist[i+1];
  }
  llist[NUM_LIST_NODES].next = &llist[0];

  int sum_gpu = 0;
  int sum_cpu = 0;

  list_head *newitem = (list_head*) malloc (sizeof(list_head));
  list_data *newdata = (list_data*) malloc (sizeof(list_data));
  newdata->data16 = 10;

  parallel_for_each(concurrency::extent<1>(1),[=, &sum_gpu](concurrency::index<1> idx) restrict(amp) {
	list_head* l = llist;
	list_insert_new(llist, newitem, newdata, NUM_LIST_NODES-1);
    for (int i = 0; i <= NUM_LIST_NODES; ++i) {
      sum_gpu += l->info->data16;
      l = l->next;
    }
  });

  llist[NUM_LIST_NODES-1].next = newitem->next;

  {
	list_head* l = llist;
	list_insert_new(llist, newitem, newdata, NUM_LIST_NODES-1);
    for (int i = 0; i <= NUM_LIST_NODES; ++i) {
      sum_cpu += l->info->data16;
      l = l->next;
    }
  }

  // verify
  int error_struct = sum_cpu - sum_gpu;
  if (error_struct == 0) {
    std::cout << "Structure Verify success!\n";
    std::cout << "sum_cpu = " << sum_cpu << " -- sum_gpu = " << sum_gpu << "\n";
  } else {
    std::cout << "Structure Verify failed!\n";
    std::cout << "sum_cpu = " << sum_cpu << " -- sum_gpu = " << sum_gpu << "\n";
  }

  free(llist);
  free(ldata);
  free(newitem);
  free(newdata);

  return (error_struct == 0);
}

int main()
{
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {
    ret &= test();
  }

  return !(ret == true);
}

