#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <raylib.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

int64_t NEAT_findNeuron(struct NEAT_Genome *g, uint32_t id) {
  int64_t index = -1;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].id == id)
      index = i;
  }
  return index;
}

uint32_t NEAT_layerNeuron(struct NEAT_Genome *g, uint32_t neuronIndex,
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

    return runningLayerCount;
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

// This code is actual garbage
void drawNetwork(struct NEAT_Genome *g, uint32_t x, uint32_t y, uint32_t w,
                 uint32_t h) {
  float radius = 0.01f * w;

  uint32_t numLayers = 0;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].kind == NEAT_NEURON_KIND_OUTPUT) {
      numLayers = g->neurons.items[i].layer + 1;
      break;
    }
  }

  uint32_t hpad = w / numLayers;
  for (uint32_t i = 0; i < numLayers; i++) {
    uint32_t vpad = 0;
    DA_CREATE(uint32_t) neuronsInLayer = { 0 };
    for (uint32_t j = 0; j < g->neurons.count; j++) {
      if (g->neurons.items[j].layer == i) {
        DA_APPEND(&neuronsInLayer, j);
      }
    }

    vpad = h / neuronsInLayer.count;

    for (uint32_t j = 0; j < neuronsInLayer.count; j++) {
      Color colour = WHITE;
      uint32_t cx1 = x + radius + i * hpad;
      uint32_t cy1 =
        (y + radius) + ((h - (neuronsInLayer.count - 1) * vpad) / 2) + j * vpad;

      DA_CREATE(uint32_t) connections = { 0 };

      for (uint32_t k = 0; k < g->connections.count; k++) {
        if (g->connections.items[k].from ==
              g->neurons.items[neuronsInLayer.items[j]].id &&
            g->connections.items[k].enabled) {
          DA_APPEND(&connections, k);
        }
      }

      DA_CREATE(uint32_t) toNeurons = { 0 };

      for (uint32_t k = 0; k < connections.count; k++) {
        DA_APPEND(
          &toNeurons,
          NEAT_findNeuron(g, g->connections.items[connections.items[k]].to));
      }

      for (uint32_t k = 0; k < toNeurons.count; k++) {
        uint32_t indexInLayer = 0;
        uint32_t neuronsInLayer2 = 0;
        for (uint32_t l = 0; l < g->neurons.count; l++) {
          if (l == toNeurons.items[k]) {
            break;
          }

          if (g->neurons.items[l].layer ==
              g->neurons.items[toNeurons.items[k]].layer) {
            indexInLayer++;
          }
        }

        for (uint32_t l = 0; l < g->neurons.count; l++) {
          if (g->neurons.items[l].layer ==
              g->neurons.items[toNeurons.items[k]].layer) {
            neuronsInLayer2++;
          }
        }
        uint32_t vpad2 = h / neuronsInLayer2;

        uint32_t cx2 =
          x + radius + g->neurons.items[toNeurons.items[k]].layer * hpad;
        uint32_t cy2 = (y + radius) +
                       ((h - (neuronsInLayer2 - 1) * vpad2) / 2) +
                       indexInLayer * vpad2;

        DrawLine(cx1, cy1, cx2, cy2, colour);
      }

      DA_FREE(&toNeurons);

      DA_FREE(&connections);

      switch (g->neurons.items[neuronsInLayer.items[j]].kind) {
        case NEAT_NEURON_KIND_INPUT:
          colour = CLITERAL(Color) RED;
          break;
        case NEAT_NEURON_KIND_BIAS:
          colour = RAYWHITE;
          break;
        case NEAT_NEURON_KIND_HIDDEN:
          colour = GREEN;
          break;
        case NEAT_NEURON_KIND_OUTPUT:
          colour = BLUE;
          break;
        default:
          assert(0 && "Unreachable");
          break;
      }

      DrawCircle(cx1, cy1, radius, colour);
    }

    DA_FREE(&neuronsInLayer);
  }

  //  exit(0);
}

void NEAT_splitConnection(struct NEAT_Genome *g, uint32_t connection,
                          struct NEAT_Context *ctx) {
  assert(connection < g->connections.count && "Invalid connection index");
  g->connections.items[connection].enabled = false;

  uint32_t maxNeuronId = 0;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    maxNeuronId = g->neurons.items[i].id > maxNeuronId ? g->neurons.items[i].id
                                                       : maxNeuronId;
  }

  NEAT_createConnection(g, maxNeuronId + 1,
                        g->connections.items[connection].from, true, ctx);

  NEAT_createConnection(g, g->connections.items[connection].to, maxNeuronId + 1,
                        true, ctx);
}

int main(void) {
  srand(time(NULL));
  struct NEAT_Context ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
    .inputs = 3,
    .outputs = 10,
    .populationSize = 7,
    .allowRecurrent = true,
    .initialSpeciesTarget = 10,
    .initialSpeciationThreshold = 1.5f,
    .improvementDeadline = 15,
  });

  //NEAT_createConnection(&ctx.population[0], 69, 3, true, &ctx);
  //NEAT_createConnection(&ctx.population[0], 6, 69, true, &ctx);

  //for (uint32_t i = 0; i < ctx.population[0].connections.count; i++) {
  //  if (ctx.population[0].connections.items[i].to == 6 &&
  //      ctx.population[0].connections.items[i].from == 3) {
  //    ctx.population[0].connections.items[i].enabled = false;
  //  }
  //}

  ////NEAT_createConnection(&ctx.population[0], 420, 69, true, &ctx);
  ////NEAT_createConnection(&ctx.population[0], 69420, 2, true, &ctx);
  ////NEAT_createConnection(&ctx.population[0], 4, 69420, true, &ctx);
  ////NEAT_createConnection(&ctx.population[0], 6, 420, false, &ctx);
  ////NEAT_createConnection(&ctx.population[0], 80085, 420, true, &ctx);
  ////NEAT_createConnection(&ctx.population[0], 6, 80085, false, &ctx);

  NEAT_splitConnection(&ctx.population[0], 0, &ctx);

  for (int i = 0; i < 10; i++) {
    NEAT_splitConnection(&ctx.population[0],
                         ctx.population[0].connections.count - 1, &ctx);
  }

  NEAT_layer(&ctx);

  for (uint32_t i = 0; i < ctx.populationSize; i++) {
    NEAT_printNetwork(&ctx.population[i]);
  }

  uint32_t factor = 80;
  uint32_t width = 16 * factor;
  uint32_t height = 9 * factor;

  InitWindow(width, height, "NEAT");
  SetWindowState(FLAG_WINDOW_RESIZABLE);

  uint32_t bgColour = 0xff181818;

  while (!WindowShouldClose()) {
    width = GetRenderWidth();
    height = GetRenderHeight();
    ClearBackground(*(Color *)&bgColour);
    BeginDrawing();
    drawNetwork(&ctx.population[0], 0.1f * width, 0.05f * height, 0.9f * width,
                0.9f * height);

    EndDrawing();
  }

  CloseWindow();

  NEAT_cleanup(&ctx);

  return 0;
}
