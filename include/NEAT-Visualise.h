#ifndef NEAT_VISUALISE_H
#define NEAT_VISUALISE_H

#include "NEAT.h"

void NEAT_Visualise_drawNetwork(struct NEAT_Genome *g, uint32_t x, uint32_t y,
                                uint32_t w, uint32_t h);

#ifdef NEAT_VISUALISE_H_IMPLEMENTATION

// This code is actual garbage
void NEAT_Visualise_drawNetwork(struct NEAT_Genome *g, uint32_t x, uint32_t y,
                                uint32_t w, uint32_t h) {
  float radius = 0.01f * w;

  uint32_t numLayers = 0;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].kind == NEAT_NEURON_KIND_OUTPUT) {
      numLayers = g->neurons.items[i].layer + 1;
      break;
    }
  }

  assert(numLayers > 1);

  uint32_t hpad = w / (numLayers - 1);
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
        float thickness = 0.005f * h;

        colour = weight >= 0 ? GREEN : RED;

        //uint8_t alpha = fabs(weight) > 1.0f ? 255 : weight * 255;

        uint8_t alpha = tanh(fabs(weight)) * 255;
        // colour = ColorAlphaBlend(
        //   colour, CLITERAL(Color){ .a = alpha, .r = 25, .g = 25, .b = 25 },
        //   WHITE);

        colour = ColorAlphaBlend(
          colour, CLITERAL(Color){ .a = alpha, .r = 35, .g = 35, .b = 35 },
          WHITE);

        if (g->connections.items[connections.items[k]].kind ==
            NEAT_CON_KIND_RECURRENT) {
          colour = weight >= 0 ? ORANGE : BLUE;

          colour = ColorAlphaBlend(
            colour, CLITERAL(Color){ .a = alpha, .r = 35, .g = 35, .b = 35 },
            WHITE);

          if (g->neurons.items[toNeurons.items[k]].layer == i) {
            if (toNeurons.items[k] == neuronsInLayer.items[j]) {
              Vector2 ringCentre = {
                .x = cx1,
                .y = cy1 - 2 * radius,
              };

              float outRad = radius * 2;
              float inRad = outRad - thickness;

              DrawRing(ringCentre, inRad, outRad, 0.0f, 360.0f, 100, colour);

            } else {
              DrawSplineBezierQuadratic(
                (Vector2[]){
                  CLITERAL(Vector2){
                    cx1 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            radius,
                    cy1 },
                  CLITERAL(Vector2){
                    cx1 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            (w * 0.05),
                    (cy1 + cy2) / 2.0f },
                  CLITERAL(Vector2){
                    cx2 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            radius,
                    cy2 },
                },
                3, thickness, colour);
            }
          } else {
            DrawLineBezier(CLITERAL(Vector2){ cx1, cy1 },
                           CLITERAL(Vector2){ cx2, cy2 }, thickness, colour);
          }

        } else {
          DrawLineEx(CLITERAL(Vector2){ cx1, cy1 },
                     CLITERAL(Vector2){ cx2, cy2 }, thickness, colour);
        }

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

#endif // NEAT_VISUALISE_H_IMPLEMENTATION

#endif // NEAT_VISUALISE_H
