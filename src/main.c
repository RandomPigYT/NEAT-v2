#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <raylib.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

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
      uint32_t cy1 = (y + radius) +
                     ((h - (neuronsInLayer.count - 1) * vpad) / 2.0f) +
                     j * vpad;

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
                       ((h - (neuronsInLayer2 - 1) * vpad2) / 2.0f) +
                       indexInLayer * vpad2;

        float weight = g->connections.items[connections.items[k]].weight;
        float thickness = fabs(0.005 * h * weight);

        colour = weight >= 0 ? GREEN : RED;

        DrawLineEx(CLITERAL(Vector2){ cx1, cy1 }, CLITERAL(Vector2){ cx2, cy2 },
                   thickness, colour);

        //DrawLine(cx1, cy1, cx2, cy2, colour);
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
}

void NEAT_mutate(struct NEAT_Genome *g) {
  // Types of mutations:
  // -> nudging the weight of a connection
  // -> randomizing the wieght of a connection
  // -> adding a connection
  // -> adding a neuron
  // -> removing a connection
  // -> removing a neuron

  (void)g;
}

int main(void) {
  srand(time(NULL));
  struct NEAT_Context ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
    .inputs = 2,
    .outputs = 1,
    .populationSize = 7,
    .allowRecurrent = true,
    .initialSpeciesTarget = 10,
    .initialSpeciationThreshold = 1.5f,
    .improvementDeadline = 15,
  });

  ctx.currentGeneration++;
  DA_FREE(&ctx.history);

  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 0, false,
                        &ctx);

  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 1, false,
                        &ctx);
  NEAT_layer(&ctx);

  for (uint32_t i = 0; i < ctx.populationSize; i++) {
    NEAT_printNetwork(&ctx.population[i]);
  }

  uint32_t factor = 120;
  uint32_t width = 16 * factor;
  uint32_t height = 9 * factor;
  SetConfigFlags(FLAG_MSAA_4X_HINT);
  InitWindow(width, height, "NEAT");
  SetWindowState(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_MAXIMIZED);

  uint32_t bgColour = 0xff181818;

  uint32_t i = 0;
  float t = 0;
  while (!WindowShouldClose()) {
    if (GetKeyPressed() == KEY_R) {
      NEAT_cleanup(&ctx);

      uint32_t inps = ((float)rand() / (float)RAND_MAX) * (5 - 1) + 1;
      uint32_t outs = ((float)rand() / (float)RAND_MAX) * (7 - 1) + 1;

      ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
        .inputs = inps,
        .outputs = outs,
        .populationSize = 7,
        .allowRecurrent = true,
        .initialSpeciesTarget = 10,
        .initialSpeciationThreshold = 1.5f,
        .improvementDeadline = 15,
      });

      i = 0;
    }
    t += GetFrameTime();

    if (t >= 1.0f && i < 10) {
      uint32_t con = rand() % ctx.population[0].connections.count;
      NEAT_splitConnection(&ctx.population[0], con, &ctx);
      NEAT_layer(&ctx);
      i++;
      t = 0.0f;
    }

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
