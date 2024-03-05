#ifndef NEAT_H
#define NEAT_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define NEAT_MAX_CON_GEN_RANGE 1.0f
#define NEAT_MIN_CON_GEN_RANGE -1.0f

#define NEAT_EPSILON 1e-6f

#define NEAT_MAX_ITERS 100

#define TO_STR__(x) #x
#define TO_STR(x) TO_STR__(x)

#define DA_CREATE(type) \
  struct {              \
    type *items;        \
    uint64_t count;     \
    uint64_t capacity;  \
  }

#define DA_INIT_CAPACITY 1
#define DA_APPEND(arr, item)                                            \
  do {                                                                  \
    if ((arr)->count >= (arr)->capacity) {                              \
      (arr)->capacity = (arr)->capacity == 0 ? DA_INIT_CAPACITY         \
                                             : 2 * (arr)->capacity;     \
      (arr)->items =                                                    \
        realloc((arr)->items, (arr)->capacity * sizeof(*(arr)->items)); \
      assert((arr)->items != NULL && "Failed to allocate memory");      \
    }                                                                   \
    (arr)->items[(arr)->count++] = (item);                              \
  } while (0)

#define DA_FREE(arr)     \
  do {                   \
    free((arr)->items);  \
    (arr)->items = NULL; \
    (arr)->count = 0;    \
    (arr)->capacity = 0; \
  } while (0)

#define DA_DELETE_ITEM(arr, index)                                            \
  do {                                                                        \
    assert((arr)->count != 0 && "Attempted to pop from empty dynamic array"); \
    assert((index) < (arr)->count && "Invalid index");                        \
    if ((index) == (arr)->count - 1) {                                        \
      (arr)->count--;                                                         \
    } else {                                                                  \
      memmove(&((arr)->items[index]), &((arr)->items[index + 1]),             \
              ((arr)->count - index - 1) * sizeof(*(arr)->items));            \
      (arr)->count--;                                                         \
    }                                                                         \
  } while (0)

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

enum NEAT_Phase {

  NEAT_COMPLEXIFY,
  NEAT_PRUNE,

};

struct NEAT_Connection {
  uint32_t from;
  uint32_t to;

  float weight;

  enum NEAT_ConnectionKind kind;

  uint64_t innovation;

  bool enabled;
};

struct NEAT_Neuron {
  // id 0 is always the bias neuron
  uint32_t id;

  enum NEAT_NeuronKind kind;
  float activation;

  uint32_t layer;
};

struct NEAT_Genome {
  DA_CREATE(struct NEAT_Connection) connections;

  DA_CREATE(struct NEAT_Neuron) neurons;
  uint32_t nextNeuronId;

  uint32_t species;

  float fitness;
};

struct NEAT_Species {
  struct NEAT_Genome representative;

  DA_CREATE(uint32_t) genomes;

  float currentFitness;

  float bestFitness;
  uint32_t gensSinceImproved;
};

struct NEAT_ConnectionRecord {
  uint32_t to;
  uint32_t from;

  enum NEAT_ConnectionKind kind;

  //uint64_t generation;

  uint64_t innovation;
};

struct NEAT_NetworkArch {
  uint32_t inputs;
  uint32_t outputs;
};

struct NEAT_Context {
  uint64_t globalInnovation;

  // Reinitialised every generation
  DA_CREATE(struct NEAT_ConnectionRecord) history;

  DA_CREATE(struct NEAT_Species) species;

  struct NEAT_NetworkArch arch;

  uint32_t populationSize;
  struct NEAT_Genome *population;

  bool allowRecurrent;
  float recurrentProbability;

  float parentMutationProbability;
  float childMutationProbability;
  uint32_t maxMutationsPerGeneration;

  float conToggleProbability;
  float weightNudgeProbability;
  float weightRandomizeProbability;
  float connectionAddProbability;
  float neuronAddProbability;
  float connectionDeleteProbability;
  float neuronDeleteProbability;

  // Elites are treated reproduced separately
  float elitismProportion;

  float asexualProportion;
  float sexualProportion;

  // Applies to sexual reproducers
  float interspeciesProbability;

  uint32_t targetSpecies;
  float speciationThreshold;

  uint32_t improvementDeadline;

  uint32_t prunePhaseThreshold;
  uint32_t pruneProbationTime;

  uint32_t currentGeneration;
};

struct NEAT_Parameters {
  uint32_t inputs;
  uint32_t outputs;
  uint32_t populationSize;

  bool allowRecurrent;
  float recurrentProbability;

  float parentMutationProbability;
  float childMutationProbability;
  uint32_t maxMutationsPerGeneration;

  float conToggleProbability;
  float weightNudgeProbability;
  float weightRandomizeProbability;
  float connectionAddProbability;
  float neuronAddProbability;
  float connectionDeleteProbability;
  float neuronDeleteProbability;

  float elitismProportion;
  float sexualProportion;
  float interspeciesProbability;

  uint32_t initialSpeciesTarget;
  float initialSpeciationThreshold;

  uint32_t improvementDeadline;

  uint32_t prunePhaseThreshold;
  uint32_t pruneProbationTime;
};

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome,
                           enum NEAT_ConnectionKind kind, uint32_t to,
                           uint32_t from, bool createMissingNeurons,
                           bool enabled, struct NEAT_Context *ctx);

struct NEAT_Genome NEAT_constructNetwork(struct NEAT_Context *ctx);

void NEAT_splitConnection(struct NEAT_Genome *g, uint32_t connection,
                          struct NEAT_Context *ctx);

struct NEAT_Context
NEAT_constructPopulation(const struct NEAT_Parameters *parameters);

bool NEAT_neuronsInterrelated(struct NEAT_Genome *g, uint32_t from,
                              uint32_t to);

void NEAT_numNeuronConnections(const struct NEAT_Genome *g, uint32_t neuron,
                               uint32_t *incoming, uint32_t *outgoing);

void NEAT_mutate(struct NEAT_Genome *g, enum NEAT_Phase phase,
                 struct NEAT_Context *ctx);

int64_t NEAT_findNeuron(const struct NEAT_Genome *g, uint32_t id);

uint32_t NEAT_layerNeuron(const struct NEAT_Genome *g, uint32_t neuronIndex,
                          uint32_t runningLayerCount);

void NEAT_layer(struct NEAT_Context *ctx);

void NEAT_cleanup(struct NEAT_Context *ctx);

void NEAT_printNetwork(const struct NEAT_Genome *g);

#ifdef NEAT_H_IMPLEMENTATION

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome,
                           enum NEAT_ConnectionKind kind, uint32_t to,
                           uint32_t from, bool createMissingNeurons,
                           bool enabled, struct NEAT_Context *ctx) {
  // Check if the connection is possible
  bool fromExists = false;
  bool toExists = false;

  for (uint64_t i = 0; i < genome->neurons.count; i++) {
    if (genome->neurons.items[i].id == from) {
      fromExists = true;
    }

    if (genome->neurons.items[i].id == to) {
      toExists = true;
    }
  }

  if ((!fromExists || !toExists) && !createMissingNeurons) {
    return false;
  }

  if (!fromExists) {
    struct NEAT_Neuron n = {
      .id = from,
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .layer = 0,
    };

    DA_APPEND(&genome->neurons, n);
  }

  if (!toExists) {
    struct NEAT_Neuron n = {
      .id = to,
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .layer = 0,
    };

    DA_APPEND(&genome->neurons, n);
  }

  bool isNovelConnection = true;
  uint64_t innovation = 0;

  for (uint64_t i = 0; i < ctx->history.count; i++) {
    struct NEAT_ConnectionRecord temp = ctx->history.items[i];
    if (temp.kind == kind && temp.from == from && temp.to == to) {
      isNovelConnection = false;
      innovation = temp.innovation;
    }
  }

  if (isNovelConnection) {
    struct NEAT_ConnectionRecord temp = {
      .from = from,
      .to = to,
      .kind = kind,
      .innovation = ctx->globalInnovation++,
    };

    innovation = temp.innovation;
    DA_APPEND(&ctx->history, temp);
  } else {
    // Check if the connection already exists in the genome
    for (uint64_t i = 0; i < genome->connections.count; i++) {
      if (genome->connections.items[i].innovation == innovation) {
        return false;
      }
    }
  }

  float weight = (((float)rand() / (float)RAND_MAX) *
                  (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                 NEAT_MIN_CON_GEN_RANGE;

  struct NEAT_Connection newConnection = {
    .from = from,
    .to = to,
    .weight = weight,
    .kind = kind,
    .innovation = innovation,
    .enabled = enabled,
  };

  DA_APPEND(&genome->connections, newConnection);

  return true;
}

struct NEAT_Genome NEAT_constructNetwork(struct NEAT_Context *ctx) {
  struct NEAT_Genome genome = {
    .connections = { 0 },
    .neurons = { 0 },
    .nextNeuronId = ctx->arch.inputs + ctx->arch.outputs,
    .species = 0,
    .fitness = 0.0f,
  };

  struct NEAT_Neuron biasNeuron = {
    .kind = NEAT_NEURON_KIND_BIAS,
    .activation = 1.0f,
    .id = 0,
    .layer = 0,
  };

  DA_APPEND(&genome.neurons, biasNeuron);

  for (uint32_t j = 1; j < ctx->arch.inputs + ctx->arch.outputs; j++) {
    struct NEAT_Neuron n = {
      .kind = NEAT_NEURON_KIND_NONE,
      .activation = 0.0f,
      .id = j,
      .layer = 0,
    };

    if (j < ctx->arch.inputs) {
      n.kind = NEAT_NEURON_KIND_INPUT;

    } else {
      n.kind = NEAT_NEURON_KIND_OUTPUT;
    }

    DA_APPEND(&genome.neurons, n);
  }

  // Fully connect the network
  for (uint32_t j = 0; j < ctx->arch.inputs; j++) {
    for (uint32_t k = ctx->arch.inputs;
         k < ctx->arch.inputs + ctx->arch.outputs; k++) {
      NEAT_createConnection(&genome, NEAT_CON_KIND_FORWARD, k, j, false, true,
                            ctx);
    }
  }

  return genome;
}

struct NEAT_Context
NEAT_constructPopulation(const struct NEAT_Parameters *parameters) {
  struct NEAT_Context ctx = {
    .history = { 0 },
		.species = { 0 },

    .arch = { 
			.inputs = parameters->inputs + 1, // Extra input for bias
      .outputs = parameters->outputs, 
		},

    .populationSize = parameters->populationSize,
    .population = NULL,

		.allowRecurrent = parameters->allowRecurrent,
		.recurrentProbability = parameters->recurrentProbability,

		.parentMutationProbability = parameters->parentMutationProbability,
		.childMutationProbability = parameters->childMutationProbability,
		.maxMutationsPerGeneration = parameters->maxMutationsPerGeneration,

		.conToggleProbability = parameters->conToggleProbability,
		.weightNudgeProbability = parameters->weightNudgeProbability,
		.weightRandomizeProbability = parameters->weightRandomizeProbability,
		.connectionAddProbability = parameters->connectionAddProbability,
		.neuronAddProbability = parameters->neuronAddProbability,
		.connectionDeleteProbability = parameters->connectionDeleteProbability,
		.neuronDeleteProbability = parameters->neuronDeleteProbability,

		.elitismProportion = parameters->elitismProportion,
		.asexualProportion = 1.0f - parameters->sexualProportion,
		.sexualProportion = parameters->sexualProportion,
		.interspeciesProbability = parameters->interspeciesProbability,

		.targetSpecies = parameters->initialSpeciesTarget,
		.speciationThreshold = parameters->initialSpeciationThreshold,
		.improvementDeadline = parameters->improvementDeadline,

		.prunePhaseThreshold = parameters->prunePhaseThreshold,
		.pruneProbationTime = parameters->pruneProbationTime,

		.currentGeneration = 0,
};

  assert(ctx.elitismProportion >= 0.0f && ctx.elitismProportion <= 1.0f);
  assert(ctx.asexualProportion >= 0.0f && ctx.asexualProportion <= 1.0f);
  assert(ctx.sexualProportion >= 0.0f && ctx.sexualProportion <= 1.0f);
  assert(ctx.interspeciesProbability >= 0.0f &&
         ctx.interspeciesProbability <= 1.0f);

  assert(ctx.parentMutationProbability >= 0.0f &&
         ctx.parentMutationProbability <= 1.0f);
  assert(ctx.childMutationProbability >= 0.0f &&
         ctx.childMutationProbability <= 1.0f);

  assert(ctx.conToggleProbability >= 0.0f);
  assert(ctx.weightNudgeProbability >= 0.0f);
  assert(ctx.weightRandomizeProbability >= 0.0f);
  assert(ctx.connectionAddProbability >= 0.0f);
  assert(ctx.neuronAddProbability >= 0.0f);
  assert(ctx.connectionDeleteProbability >= 0.0f);
  assert(ctx.neuronDeleteProbability >= 0.0f);

  ctx.population = malloc(ctx.populationSize * sizeof(struct NEAT_Genome));
  assert(ctx.population != NULL && "Failed to allocate memory for population");

  for (uint32_t i = 0; i < ctx.populationSize; i++) {
    ctx.population[i] = NEAT_constructNetwork(&ctx);
  }

  return ctx;
}

bool NEAT_neuronsInterrelated(struct NEAT_Genome *g, uint32_t from,
                              uint32_t to) {
  bool neuronsInterrelated = false;
  DA_CREATE(uint32_t) neuronQueue = { 0 };
  DA_APPEND(&neuronQueue, from);

  DA_CREATE(uint32_t) checkedNeurons = { 0 };
  while (neuronQueue.count) {
    uint32_t n = neuronQueue.items[0];
    DA_DELETE_ITEM(&neuronQueue, 0);

    DA_APPEND(&checkedNeurons, n);

    if (n == to) {
      neuronsInterrelated = true;
      break;
    }

    for (uint32_t i = 0; i < g->connections.count; i++) {
      if (g->connections.items[i].kind == NEAT_CON_KIND_FORWARD &&
          g->connections.items[i].to == g->neurons.items[n].id) {
        uint32_t temp = NEAT_findNeuron(g, g->connections.items[i].from);

        bool notChecked = true;
        for (uint32_t j = 0; j < checkedNeurons.count; j++) {
          if (checkedNeurons.items[j] == temp) {
            notChecked = false;
            break;
          }
        }

        if (notChecked)
          DA_APPEND(&neuronQueue, temp);
      }
    }
  }

  DA_FREE(&neuronQueue);
  DA_FREE(&checkedNeurons);

  return neuronsInterrelated;
}

void NEAT_numNeuronConnections(const struct NEAT_Genome *g, uint32_t neuron,
                               uint32_t *incoming, uint32_t *outgoing) {
  assert(neuron < g->neurons.count && "Invalid neuron index");

  *incoming = 0;
  *outgoing = 0;

  uint32_t neuronId = g->neurons.items[neuron].id;
  for (uint32_t i = 0; i < g->connections.count; i++) {
    if (g->connections.items[i].to == neuronId) {
      (*incoming)++;
    }

    if (g->connections.items[i].from == neuronId) {
      (*outgoing)++;
    }
  }
}

void NEAT_mutate(struct NEAT_Genome *g, enum NEAT_Phase phase,
                 struct NEAT_Context *ctx) {
  // Types of mutations:
  // -> nudging the weight of a connection
  // -> randomizing the wieght of a connection
  // -> adding a connection
  // -> adding a neuron
  // -> removing a connection
  // -> removing a neuron

  float conToggleProb = ctx->conToggleProbability;
  float nudgeProb = ctx->weightNudgeProbability;
  float randProb = ctx->weightRandomizeProbability;
  float conStructureModProb = phase == NEAT_COMPLEXIFY
                                ? ctx->connectionAddProbability
                                : ctx->connectionDeleteProbability;
  float neuronStructureModProb = phase == NEAT_COMPLEXIFY
                                   ? ctx->neuronAddProbability
                                   : ctx->neuronDeleteProbability;

  if (g->connections.count == 0) {
    if (phase == NEAT_PRUNE) {
      return;
    }

    conStructureModProb = 1.0f;

    neuronStructureModProb = 0.0f;
    randProb = 0.0f;
    nudgeProb = 0.0f;
    conToggleProb = 0.0f;
  }

  // Check if there are no hidden neurons during the purning phase
  if (g->neurons.count == ctx->arch.inputs + ctx->arch.outputs &&
      phase == NEAT_PRUNE) {
    //nudgeProb += neuronStructureModProb / 4;
    //randProb += neuronStructureModProb / 4;
    //conStructureModProb += neuronStructureModProb / 4;
    //conToggleProb += neuronStructureModProb / 4;

    neuronStructureModProb = 0.0f;
  }

  DA_CREATE(uint32_t) removableNeurons = { 0 };

  // Check if there are no neurons with at least one end that is only connected to one connection
  if (phase == NEAT_PRUNE) {
    for (uint32_t i = 0; i < g->neurons.count; i++) {
      if (g->neurons.items[i].kind != NEAT_NEURON_KIND_HIDDEN) {
        continue;
      }

      uint32_t numIncoming = 0;
      uint32_t numOutgoing = 0;

      NEAT_numNeuronConnections(g, i, &numIncoming, &numOutgoing);

      if (numIncoming == 1 || numOutgoing == 1 || numIncoming == 0 ||
          numOutgoing == 0) {
        DA_APPEND(&removableNeurons, i);
      }
    }

    if (removableNeurons.count == 0) {
      //nudgeProb += neuronStructureModProb / 4;
      //randProb += neuronStructureModProb / 4;
      //conStructureModProb += neuronStructureModProb / 4;
      //conToggleProb += neuronStructureModProb / 4;

      neuronStructureModProb = 0.0f;
    }
  }

  float runningSum = 0.0f;
  float random = (float)rand() / (float)RAND_MAX;

  runningSum += nudgeProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToNudge = rand() % g->connections.count;

    float amount = (((float)rand() / (float)RAND_MAX) *
                    (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                   NEAT_MIN_CON_GEN_RANGE;

    g->connections.items[weightToNudge].weight += amount;

    goto Exit;
  }

  runningSum += randProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToRandomize = rand() % g->connections.count;

    float newWeight = (((float)rand() / (float)RAND_MAX) *
                       (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                      NEAT_MIN_CON_GEN_RANGE;

    g->connections.items[weightToRandomize].weight = newWeight;

    goto Exit;
  }

  runningSum += conToggleProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToToggle = rand() % g->connections.count;

    g->connections.items[weightToToggle].enabled =
      !g->connections.items[weightToToggle].enabled;

    goto Exit;
  }

  runningSum += conStructureModProb;
  if (random < runningSum) {
    if (phase == NEAT_COMPLEXIFY) {
      bool isRecurrent = false;
      if (ctx->allowRecurrent) {
        float random = (float)rand() / (float)RAND_MAX;
        isRecurrent = random < ctx->recurrentProbability;
      }

      if (isRecurrent) {
        uint32_t n1 = rand() % g->neurons.count;
        uint32_t n2 = rand() % g->neurons.count;

        uint32_t fromId = g->neurons.items[n1].id;
        uint32_t toId = g->neurons.items[n2].id;

        //if (g->neurons.items[n2].layer > g->neurons.items[n1].layer) {
        //  uint32_t temp = fromId;
        //  fromId = toId;
        //  toId = temp;
        //}

        NEAT_createConnection(g, NEAT_CON_KIND_RECURRENT, toId, fromId, false,
                              true, ctx);

      } else {
        bool neuronsInterrelated;

        uint32_t to, from;

        uint32_t iter = 0;
        do {
          to = rand() % g->neurons.count;
          from = rand() % g->neurons.count;

          neuronsInterrelated = NEAT_neuronsInterrelated(g, from, to);

        } while ((to == from || neuronsInterrelated ||
                  g->neurons.items[to].kind == NEAT_NEURON_KIND_INPUT ||
                  g->neurons.items[to].kind == NEAT_NEURON_KIND_BIAS ||
                  g->neurons.items[from].kind == NEAT_NEURON_KIND_OUTPUT) &&
                 iter++ < NEAT_MAX_ITERS);

        if (iter == NEAT_MAX_ITERS) {
          goto Exit;
        }

        NEAT_createConnection(g, NEAT_CON_KIND_FORWARD, g->neurons.items[to].id,
                              g->neurons.items[from].id, false, true, ctx);
      }

      goto Exit;

    } else {
      assert(g->connections.count > 0);

      uint32_t conToRemove = rand() % g->connections.count;

      DA_DELETE_ITEM(&g->connections, conToRemove);
      goto Exit;
    }
  }

  runningSum += neuronStructureModProb;
  if (random < runningSum) {
    if (phase == NEAT_COMPLEXIFY) {
      assert(g->connections.count != 0);

      DA_CREATE(uint32_t) enabledCons = { 0 };
      for (uint32_t i = 0; i < g->connections.count; i++) {
        if (g->connections.items[i].enabled) {
          DA_APPEND(&enabledCons, i);
        }
      }

      if (enabledCons.count == 0) {
        DA_FREE(&enabledCons);
        goto Exit;
      }

      uint32_t conToSplit = enabledCons.items[rand() % enabledCons.count];

      g->connections.items[conToSplit].enabled = false;
      enum NEAT_ConnectionKind kind = g->connections.items[conToSplit].kind;

      // Implicitly creates the new node
      NEAT_createConnection(g, kind, g->nextNeuronId,
                            g->connections.items[conToSplit].from, true, true,
                            ctx);

      NEAT_createConnection(g, kind, g->connections.items[conToSplit].to,
                            g->nextNeuronId, false, true, ctx);

      g->nextNeuronId++;

      DA_FREE(&enabledCons);
      goto Exit;

    } else {
      assert(removableNeurons.count != 0);

      uint32_t neuronToRemove =
        removableNeurons.items[rand() % removableNeurons.count];

      uint32_t neuronId = g->neurons.items[neuronToRemove].id;

      uint32_t numIncoming, numOutgoing;
      NEAT_numNeuronConnections(g, neuronToRemove, &numIncoming, &numOutgoing);

      if (numIncoming == 0 || numOutgoing == 0) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId ||
              g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
          }
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        DA_FREE(&conIdsToRemove);

      } else if (numIncoming == 1) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices
        DA_CREATE(uint32_t) outgoingNeuronIds = { 0 };

        uint32_t incomingNeuronId = 0;
        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId) {
            incomingNeuronId = g->connections.items[i].from;
            break;
          }
        }

        // Connections that originate from the target neuron
        DA_CREATE(uint32_t) outgoingCons = { 0 };
        uint32_t incomingCon = 0;

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            incomingCon = i;

          } else if (g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            DA_APPEND(&outgoingNeuronIds, g->connections.items[i].to);
            DA_APPEND(&outgoingCons, i);
          }
        }

        for (uint32_t i = 0; i < outgoingNeuronIds.count; i++) {
          if (incomingNeuronId == neuronId ||
              outgoingNeuronIds.items[i] == neuronId) {
            continue;
          }

          enum NEAT_ConnectionKind kind =
            g->connections.items[outgoingCons.items[i]].kind;

          if (kind != g->connections.items[incomingCon].kind) {
            continue;
          }

          if (kind == NEAT_CON_KIND_FORWARD &&
              incomingNeuronId == outgoingNeuronIds.items[i]) {
            continue;
          }

          uint32_t incomingNeuron = NEAT_findNeuron(g, incomingNeuronId);
          uint32_t outgoingNeuron =
            NEAT_findNeuron(g, outgoingNeuronIds.items[i]);
          if (kind == NEAT_CON_KIND_FORWARD &&
              NEAT_neuronsInterrelated(g, incomingNeuron, outgoingNeuron)) {
            continue;
          }

          bool enabled = g->connections.items[outgoingCons.items[i]].enabled &&
                         g->connections.items[incomingCon].enabled;

          NEAT_createConnection(g, kind, outgoingNeuronIds.items[i],
                                incomingNeuronId, false, enabled, ctx);
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        DA_FREE(&outgoingCons);
        DA_FREE(&conIdsToRemove);
        DA_FREE(&outgoingNeuronIds);

      } else if (numOutgoing == 1) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices
        DA_CREATE(uint32_t) incomingNeuronIds = { 0 };

        uint32_t outgoingNeuronId = 0;
        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].from == neuronId) {
            outgoingNeuronId = g->connections.items[i].to;
            break;
          }
        }

        // Connections that originate from the target neuron
        DA_CREATE(uint32_t) incomingCons = { 0 };
        uint32_t outgoingCon = 0;

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            outgoingCon = i;

          } else if (g->connections.items[i].to == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            DA_APPEND(&incomingNeuronIds, g->connections.items[i].from);
            DA_APPEND(&incomingCons, i);
          }
        }

        for (uint32_t i = 0; i < incomingNeuronIds.count; i++) {
          if (outgoingNeuronId == neuronId ||
              incomingNeuronIds.items[i] == neuronId) {
            continue;
          }

          enum NEAT_ConnectionKind kind =
            g->connections.items[incomingCons.items[i]].kind;

          if (kind != g->connections.items[outgoingCon].kind) {
            continue;
          }

          if (kind == NEAT_CON_KIND_FORWARD &&
              outgoingNeuronId == incomingNeuronIds.items[i]) {
            continue;
          }

          uint32_t incomingNeuron =
            NEAT_findNeuron(g, incomingNeuronIds.items[i]);
          uint32_t outgoingNeuron = NEAT_findNeuron(g, outgoingNeuronId);
          if (kind == NEAT_CON_KIND_FORWARD &&
              NEAT_neuronsInterrelated(g, incomingNeuron, outgoingNeuron)) {
            continue;
          }

          bool enabled = g->connections.items[incomingCons.items[i]].enabled &&
                         g->connections.items[outgoingCon].enabled;

          NEAT_createConnection(g, kind, outgoingNeuronId,
                                incomingNeuronIds.items[i], false, enabled,
                                ctx);
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        DA_FREE(&incomingCons);
        DA_FREE(&conIdsToRemove);
        DA_FREE(&incomingNeuronIds);
      }

      DA_DELETE_ITEM(&g->neurons, neuronToRemove);

      goto Exit;
    }
  }

Exit:
  DA_FREE(&removableNeurons);
  return;
}

void NEAT_splitConnection(struct NEAT_Genome *g, uint32_t connection,
                          struct NEAT_Context *ctx) {
  assert(connection < g->connections.count && "Invalid connection index");
  g->connections.items[connection].enabled = false;

  //uint32_t maxNeuronId = 0;
  //for (uint32_t i = 0; i < g->neurons.count; i++) {
  //  maxNeuronId = g->neurons.items[i].id > maxNeuronId ? g->neurons.items[i].id
  //                                                     : maxNeuronId;
  //}

  enum NEAT_ConnectionKind kind = g->connections.items[connection].kind;

  NEAT_createConnection(g, kind, g->nextNeuronId,
                        g->connections.items[connection].from, true, true, ctx);

  NEAT_createConnection(g, kind, g->connections.items[connection].to,
                        g->nextNeuronId, true, true, ctx);

  g->nextNeuronId++;
}

void NEAT_printNetwork(const struct NEAT_Genome *g) {
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    const char *kind;
    switch (g->neurons.items[i].kind) {
      case NEAT_NEURON_KIND_INPUT:
        kind = "input";
        break;
      case NEAT_NEURON_KIND_OUTPUT:
        kind = "output";
        break;
      case NEAT_NEURON_KIND_BIAS:
        kind = "bias";
        break;
      case NEAT_NEURON_KIND_HIDDEN:
        kind = "hidden";
        break;

      default:
        assert(0 && "Unreachable");
        break;
    }

    printf("id: %d\tkind: %s\tlayer: %d\tactivation: %f\n",
           g->neurons.items[i].id, kind, g->neurons.items[i].layer,
           g->neurons.items[i].activation);
  }

  for (uint32_t i = 0; i < g->connections.count; i++) {
    const char *kind;
    switch (g->connections.items[i].kind) {
      case NEAT_CON_KIND_FORWARD:
        kind = "FWD";
        break;
      case NEAT_CON_KIND_RECURRENT:
        kind = "REC";
        break;
      default:
        assert(0 && "Unreachable");
        break;
    }
    printf(
      "innovation: %ld\tfrom: %d\tto: %d\tweight: %f\tkind: %s\tenabled: %s\n",
      g->connections.items[i].innovation, g->connections.items[i].from,
      g->connections.items[i].to, g->connections.items[i].weight, kind,
      g->connections.items[i].enabled ? "true" : "false");
  }
}

int64_t NEAT_findNeuron(const struct NEAT_Genome *g, uint32_t id) {
  int64_t index = -1;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].id == id)
      index = i;
  }
  return index;
}

uint32_t NEAT_layerNeuron(const struct NEAT_Genome *g, uint32_t neuronIndex,
                          uint32_t runningLayerCount) {
  if (g->neurons.items[neuronIndex].kind == NEAT_NEURON_KIND_INPUT ||
      g->neurons.items[neuronIndex].kind == NEAT_NEURON_KIND_BIAS) {
    return runningLayerCount;
  }

  DA_CREATE(uint32_t) inConnections = { 0 };

  for (uint32_t i = 0; i < g->connections.count; i++) {
    if (g->connections.items[i].enabled &&
        g->connections.items[i].kind == NEAT_CON_KIND_FORWARD &&
        g->connections.items[i].to == g->neurons.items[neuronIndex].id) {
      DA_APPEND(&inConnections, i);
    }
  }

  if (!inConnections.count) {
    if (runningLayerCount == 0)
      return 1;

    return runningLayerCount + 1;
  }

  uint32_t maxLayer = 0;
  for (uint32_t i = 0; i < inConnections.count; i++) {
    int64_t neuronIndex =
      NEAT_findNeuron(g, g->connections.items[inConnections.items[i]].from);
    assert(neuronIndex >= 0 && "invalid connection");

    uint32_t temp = NEAT_layerNeuron(g, neuronIndex, runningLayerCount + 1);
    maxLayer = temp > maxLayer ? temp : maxLayer;
  }

  DA_FREE(&inConnections);
  return maxLayer;
}

void NEAT_layer(struct NEAT_Context *ctx) {
  for (uint32_t i = 0; i < ctx->populationSize; i++) {
    struct NEAT_Genome *g = &ctx->population[i];

    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      g->neurons.items[neuron].layer = NEAT_layerNeuron(g, neuron, 0);
    }

    uint32_t maxLayer = 0;
    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      if (g->neurons.items[neuron].kind != NEAT_NEURON_KIND_OUTPUT) {
        maxLayer = g->neurons.items[neuron].layer > maxLayer
                     ? g->neurons.items[neuron].layer
                     : maxLayer;
      }
    }

    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      if (g->neurons.items[neuron].kind == NEAT_NEURON_KIND_OUTPUT) {
        g->neurons.items[neuron].layer = maxLayer + 1;
      }

      if (g->neurons.items[neuron].layer == 0 &&
          (g->neurons.items[neuron].kind != NEAT_NEURON_KIND_INPUT &&
           g->neurons.items[neuron].kind != NEAT_NEURON_KIND_BIAS)) {
        g->neurons.items[neuron].layer = 1;
      }
    }
  }
}

void NEAT_cleanup(struct NEAT_Context *ctx) {
  for (uint32_t i = 0; i < ctx->populationSize; i++) {
    DA_FREE(&ctx->population[i].connections);
    DA_FREE(&ctx->population[i].neurons);
  }

  DA_FREE(&ctx->history);

  free(ctx->population);
}
#endif // NEAT_H_IMPLEMENTATION

#endif // NEAT_H
