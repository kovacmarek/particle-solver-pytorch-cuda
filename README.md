# Particle solver inside houdini using GPUs & Pytorch

## This solver is only the preview.Further iterations will include significant clean up of the code, VDB collision support and stable particle behaviour.

### Liquid behaviour of particles with surface tension

![liquid](https://user-images.githubusercontent.com/30508711/211037554-7d24335a-8886-4e4f-9a00-0133a2bd16c9.gif)

### Splitting simulation into multiple bounding boxes allows for higher point count with same amount of VRAM. Additionaly, this opens up possibilities for multi-GPU compute, since each section can be run on a separate device in parallel, also expanding total VRAM capacity.

![bounding_box_particles](https://user-images.githubusercontent.com/30508711/211037920-1069e944-7092-48fa-8e02-a4d2e49e5aba.gif)

### Collision detection is done using only the polygon's centroid and its Normal, instead of vertices.

![photo_2022-01-12_22-51-20 (2)](https://user-images.githubusercontent.com/30508711/211038134-0c634509-e867-492f-af14-f8f5e6cd70d0.jpg)

### Collision detection example

![collision_particles_V2](https://user-images.githubusercontent.com/30508711/211042445-9d9fe7c4-70fd-4ecc-86a7-007a3c4be768.gif)
