#ifndef NEAT_H
#define NEAT_H

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#define TO_STR__(x) #x
#define TO_STR(x) TO_STR__(x)

#define DA_CREATE(type) \
  struct {              \
    type *items;        \
    uint64_t count;     \
    uint64_t capacity;  \
  }

#define DA_INIT_CAPACITY 25

#define DA_APPEND(arr, item)                                                \
  {                                                                         \
    if ((arr)->count >= (arr)->capacity) {                                  \
      (arr)->capacity = (arr)->capacity == 0 ? DA_INIT_CAPACITY             \
                                             : 2 * (arr)->capacity;         \
      (arr)->items =                                                        \
        realloc((arr)->items, (arr)->capacity * sizeof(*(arr)->items));     \
      assert((arr)->items != NULL &&                                        \
             (__FILE__ ":" TO_STR(__LINE__) " Failed to allocate memory")); \
    }                                                                       \
    (arr)->items[(arr)->count++] = (item);                                  \
  }

enum NEAT_NeuronKind {

  NEAT_NEURON_KIND_NONE = 0,
  NEAT_NEURON_KIND_INPUT,
  NEAT_NEURON_KIND_OUTPUT,
  NEAT_NEURON_KIND_BIAS,
  NEAT_NEURON_KIND_HIDDEN,

};

enum NEAT_ActivationFunction {

  NEAT_ACTIVATION_NONE = 0,

  NEAT_ACTIVATION_SIGMOID,
  NEAT_ACTIVATION_TANH,
  NEAT_ACTIVATION_RELU,
  NEAT_ACTIVATION_LEAKY_RELU,

  /* ... */

};

enum NEAT_ConnectionKind {

  NEAT_CON_KIND_NONE = 0,

  NEAT_CON_KIND_FORWARD,
  NEAT_CON_KIND_RECURRENT,

};

struct NEAT_Connection {
  uint64_t to;
  uint64_t from;

  float weight;

  uint64_t innovation;

  bool enabled;
};

struct NEAT_Neuron {
  uint64_t ID;
  enum NEAT_NeuronKind kind;
};

struct NEAT_Genome {
  DA_CREATE(struct NEAT_Connection) connections;
  DA_CREATE(struct NEAT_Neuron) neurons;

  uint64_t species;
};

struct NEAT_ConnectionRecord {
  uint64_t to;
  uint64_t from;

  uint64_t innovation;
};

struct NEAT_Context {
  uint64_t innovation;
  DA_CREATE(struct NEAT_ConnectionRecord) history;
};

#ifdef NEAT_H_IMPLEMENTATION

// Library stuff goes here

#endif // NEAT_H_IMPLEMENTATION

#endif // NEAT_H
