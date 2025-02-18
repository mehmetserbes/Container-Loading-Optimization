# Container Loading Optimization Project

## Overview

The Container Loading Optimization Project is designed to efficiently arrange items (such as pipes) within a container of fixed dimensions. Using advanced mathematical modeling and Mixed-Integer Linear Programming (MILP) techniques, the project optimizes the placement of items by considering various constraints such as stacking, nesting, and non-overlapping in a three-dimensional space. This leads to improved space utilization, reduced shipping costs, and enhanced overall logistics efficiency.

## Features

- **3D Container Loading:** Models the loading process in three dimensions (X, Y, and Z), ensuring that items do not overlap.
- **Nested & Stacking Arrangements:** Supports both nested and stacking placements of items, allowing for more compact arrangements.
- **Physical Constraints:** Incorporates constraints related to item dimensions, container boundaries, and stacking rules.
- **Visualization:** Uses Matplotlib to visualize the container’s vertical cross-section with realistic cylindrical representations of the items.
- **Customizable Parameters:** Easily modify container dimensions, item properties (e.g., diameter, height, insulation thickness), and other parameters to suit different use cases.

## Installation

### Prerequisites

- Python 3.6 or later
- [PuLP](https://coin-or.github.io/pulp/) (for optimization)
- [Matplotlib](https://matplotlib.org/) (for visualization)

### Installing Dependencies

You can install the required Python packages using pip:

```
pip install pulp matplotlib
```

## Usage

1. **Clone the Repository:**

   ```
   git clone https://github.com/yourusername/container-loading-optimization.git
   cd container-loading-optimization
   ```

2. **Run the Model:**

   Execute the main Python script:

   ```
   python container_loading.py
   ```

   This script sets up the MILP model, solves the container loading problem, and displays a visualization of the container with the loaded items represented as cylinders.

## Model Description

The project leverages a mathematical model that includes:

- **Decision Variables:**
  - Assignment variables for items to containers.
  - Level variables indicating whether an item is placed as outer, nested level-2, or nested level-3.
  - Nested assignment variables to decide the host container for nested items.
  - Stacking variables to manage stacking (either directly on the container floor or on top of another item).
  - 3D position variables (X, Y, Z) for outer items.
  - Non-overlap and directional separation variables to avoid item overlap in the X-Y plane.

- **Constraints:**
  - Each item is assigned to exactly one container and one level.
  - Physical constraints ensure that items are placed within container boundaries.
  - Nested and stacking constraints ensure proper arrangement of items without overlapping.
  - A specific “outerPair” mechanism ensures that the non-overlap constraints are applied consistently to all outer items.

- **Objective Function:**
  - The primary objective is to minimize the number of containers used, which directly translates to cost and space efficiency.

## Contributing

Contributions to the project are welcome! If you have ideas or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact serbes.mehmet26@gmail.com.


