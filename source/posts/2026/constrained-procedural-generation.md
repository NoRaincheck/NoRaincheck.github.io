---
title: Constrained Procedural Generation
date: 2026-05-01
tags: ["Python", "procedural-generation"]
---

## Constrained Procedural Generation

I've always been interested in procedural generation, the maps and algorithms to
make 'realistic' and 'dynamic' environments. This post is more to post some
musings and experiments.

One thing that I've found is that its often difficult to _constrain_ the
generation. For example if I want to use diamond-square, then the map itself
needs to be square shape. How might we constraint or have a sliding window
approach to generate non-square areas?

The easiest way is "seed" the boundaries of the area you are generating so that
the algorithm won't violate it. There are still some artifacts from this
approach (if there are 'interesting' distributions of values from the initial
seed it may 'carry over' - there are probably regeneration approaches that can
'disguise' this)

Then to add more variation or smoothing, I found Voronoi approach suits it well,
leading to areas that are 'clumped' together to represent a 'cell' of common
ground. This can then lead to the map be tiled (e.g. a hex map) that can be used
in a TTRPG setup. I found re-applying Voronoi after diamond-square also provides
the opportunity for the map to re-normalise itself (e.g. you can normalise each
cell) that can also help with the distribution of heightmap data (especially if
you wish to apply certain constraints to it).

| Constraint                            | Mechanism                                                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Ocean at edges**                    | Diamond-square border seeding (`-2 * roughness`) + edge falloff (pre/post-Voronoi) + pinned Voronoi border seeds |
| **Water ratio**                       | Binary search on gamma power to hit target percentage below threshold                                            |
| **Regional coherence**                | Voronoi cell averaging (82% regional mean + 18% local detail)                                                    |
| **Natural-looking seed distribution** | Lloyd's relaxation moves seeds to Voronoi centroids                                                              |
| **Border seeds stay on border**       | Border points are excluded from relaxation (only inner points move)                                              |
| **No spurious inland coasts**         | Flood-fill analysis reclassifies inland coast components                                                         |
| **Rivers reach water**                | Downhill trace validates path ends in ocean/coast                                                                |
| **Rivers don't loop**                 | Seen-set prevents cycles; max iteration cap prevents infinite loops                                              |

If you want to see it in action, I have it in pure stdlib Python in my new
repository: https://github.com/NoRaincheck/kitnega where I'm developing personal
patterns using purely stdlib Python (which mostly avoids issues with supply
chain attacks - though of course with severe downsides of reinventing the wheel)
