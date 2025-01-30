# Pathfinding Algorithms for London Underground Optimization

## About This Project
This project implements various pathfinding algorithms to optimize shortest routes within the London Underground network. Algorithms used include A*, BFS, DFS, UCS, and heuristic search to determine the most efficient travel routes based on station connections, zones, and travel times.

## Built With
- Python 3.x
- pandas
- numpy
- networkx
- matplotlib

## Dataset
The project uses the **London Underground Station Connections Dataset**. Ensure the `london_underground_data.csv` file is in the root directory. The dataset includes:
- **Station Names**: List of stations
- **Zones**: Corresponding zones of each station
- **Travel Times**: Travel time between connected stations
- **Tube Lines**: Tube lines connecting stations

## Algorithms Implemented
1. A* (A-star)
2. Breadth-First Search (BFS)
3. Depth-First Search (DFS)
4. Uniform Cost Search (UCS)
5. Heuristic BFS

## Features
- Optimizes shortest route based on station connections and travel times.
- Compares computational efficiency of each algorithm.
- Visualizes the pathfinding results on a London Underground map.
- Measures the performance of algorithms based on travel time and computational time.

## How to Run
### Prerequisites
- Python 3.x
- Required libraries:
  ```bash
  pip install pandas numpy networkx matplotlib
