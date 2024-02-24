#ifndef NEAT_H
#define NEAT_H

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#define NEAT_MAX_CON_GEN_RANGE 1.0f
#define NEAT_MIN_CON_GEN_RANGE -1.0f

#define TO_STR__(x) #x
#define TO_STR(x) TO_STR__(x)

#define DA_CREATE(type) \
  struct {              \
    type *items;        \
    uint64_t count;     \
    uint64_t capacity;  \
  }

#define DA_INIT_CAPACITY 25

#define DA_APPEND(arr, item)                                            \
  {                                                                     \
    if ((arr)->count >= (arr)->capacity) {                              \
      (arr)->capacity = (arr)->capacity == 0 ? DA_INIT_CAPACITY         \
                                             : 2 * (arr)->capacity;     \
      (arr)->items =                                                    \
        realloc((arr)->items, (arr)->capacity * sizeof(*(arr)->items)); \
      assert((arr)->items != NULL && " Failed to allocate memory");     \
    }                                                                   \
    (arr)->items[(arr)->count++] = (item);                              \
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
  uint32_t from;
  uint32_t to;

  float weight;

  uint64_t innovation;

  bool enabled;
};

struct NEAT_Neuron {
  enum NEAT_NeuronKind kind;
  float activation;

  uint32_t id;
};

struct NEAT_Genome {
  DA_CREATE(struct NEAT_Connection) connections;
  DA_CREATE(struct NEAT_Neuron) neurons;

  uint32_t species;
};

struct NEAT_Species {
  DA_CREATE(struct NEAT_Genome *) genomes;

  float currentFitness;

  float bestFitness;
  uint32_t gensSinceImproved;
};

struct NEAT_ConnectionRecord {
  uint32_t to;
  uint32_t from;

  uint64_t generation;
};

struct NEAT_Context {
  // The index of the record determines the innovation
  // DO NOT CHANGE ITS ORDER!
  DA_CREATE(struct NEAT_ConnectionRecord) history;

  uint32_t populationSize;
  struct NEAT_Genome *population;

  uint32_t targetSpecies;
  float speciationThreshold;

  uint32_t currentGeneration;
};

#ifdef NEAT_H_IMPLEMENTATION

// Library stuff goes here

#endif // NEAT_H_IMPLEMENTATION

#endif // NEAT_H
