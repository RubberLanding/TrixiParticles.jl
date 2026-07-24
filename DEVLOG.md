## DEVLOG - Branch: Refinement_ParticleSpacing

## 27.05.26
- [] TODO: Update the `Refinement_ParticleSpacing` branch to not use the particle spacing explitely and update the smoothing length instead 

## 25.05.26
- [] Introduced `ResizeBuffer`, update unit tests and check code validity

## 24.05.26
- Implement `resize!()` on the CPU for now (TODO: Implement on GPU)

## 23.05.26
- Assume that `resize!()` is properly working when adding or removing particles
- Implement buffer approach later. 
- Remember to update
    - [] merge branch 
    - [] split branch 
- Test with new branch that combines the resize and the merge/split branches
- Remember to reset all the `_candidate`arrays before calling split and merge 

## 19.05.26
- Do not directly merge the resize branch into the base branch
- Keep them independent and test when putting everything together 

## 18.05.26
- Test `merge_particles_inner!()`:
    - Create a fluid system with custom spaced particles. 
    - Select some particles and their neighbors. 
    - Set the reference mass of these to be higher than the sum of the particle and neighbor mass.
    - Check if the particle and its closest neighbor got merged:
        - Where the higher index got absorbed by the lower index particle.
        - Check if the properties got correctly copied.

## 19.05.26
- Do not directly merge the resize branch into the base branch
- Keep them independent and test when putting everything together 

## 18.05.26
- Test `merge_particles_inner!()`:
    - Create a fluid system with custom spaced particles. 
    - Select some particles and their neighbors. 
    - Set the reference mass of these to be higher than the sum of the particle and neighbor mass.
    - Check if the particle and its closest neighbor got merged:
        - Where the higher index got absorbed by the lower index particle.
        - Check if the properties got correctly copied.

## 14.02.26
- Checked the paper again, I dont quite see why we need the `set_refinement_spacing` code. 
- So I decided to skip writing a unit test for this till I know when I actually need it. 
