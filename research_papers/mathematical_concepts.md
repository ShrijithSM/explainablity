# Core Mathematical Concepts for the Chronoscope System

1. **Structural Causal Models & Interventions (Pearl's Causal Inference)**
   - **Concept**: Activation Patching (or Causal Scrubbing). Represents neural reasoning traces as a directed acyclic graph.
   - **Relevance**: To prove causality, we swap activations from a corrupted run into a clean run to measure "Interchange Intervention Effect" (IIE).

2. **Dynamical Systems (Dynamic Time Warping - DTW)**
   - **Concept**: Algorithms that measure similarity between two temporal sequences that vary in speed.
   - **Relevance**: LLM CoT traces vary in token length. DTW aligns the multivariable residual stream trajectories so you can compare a 'valid' trace against an 'ablated' trace point-to-point.

3. **Topological Data Analysis (Persistent Homology)**
   - **Concept**: Studies the shape of data. Uses Betti numbers to measure connected components, holes, and voids in high-dimensional manifolds.
   - **Relevance**: To map the "smoothness" and continuity of a reasoning trace. A logical hallucination might represent a discontinuous break (a topological hole) in the residual stream manifold.

4. **Information Geometry (Fisher Information & SVD)**
   - **Concept**: Measuring the geometry of probability distributions. Applied via Singular Value Decomposition on Attention matrices.
   - **Relevance**: Used to compress high-dimensional states and measure how much "information" about the structural constraint (e.g., physical boundary) is retained across the sequence.

---
*Related papers have been downloaded to this directory.*
