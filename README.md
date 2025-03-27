# <img src="docs/images/preview.png" width="32" height="32" alt="Icon" align="left"> TornadoVM Bytecode Analyzer

A powerful visualization and analysis tool for TornadoVM bytecode logs. This tool helps developers understand task graph dependencies, memory operations, and execution patterns in TornadoVM applications.

![TornadoVM Bytecode Analyzer](docs/images/img.png)
## Features

### 1. Task Graph Analysis
- Interactive visualization of task graph dependencies
- Detailed analysis of task relationships and data flow
- Color-coded nodes and edges for better understanding
- Support for large-scale task graphs with automatic layout optimization

### 2. Memory Operation Analysis
- Track memory allocations and deallocations
- Visualize memory usage patterns over time
- Monitor object persistence and lifecycle
- Analyze memory transfer operations between host and device

### 3. Bytecode Details
- Comprehensive view of all bytecode operations
- Filtering and search capabilities
- Detailed operation information including:
  - Operation types
  - Object references
  - Memory sizes
  - Task names
  - Status information

### 4. Interactive Dashboard
- Real-time metrics and statistics
- Summary views of task graphs and operations
- Memory usage charts and distribution graphs
- Object persistence analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Using pip

```bash
pip install tornadovm-bytecode-analyzer
```

### Option 2: From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tornadovm-bytecode-analyzer.git
cd tornadovm-bytecode-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Analyzer

```bash
streamlit run tornado-visualizer-fixed.py
```

This will start the web interface, typically accessible at `http://localhost:8501`.

### Analyzing Bytecode Logs

1. Generate a bytecode log from your TornadoVM application:
```bash
tornado --printBytecodes > bytecode.log
```

2. In the web interface:
   - Upload your bytecode log file using the sidebar
   - Navigate through different analysis views:
     - Basic Overview
     - Task Graphs
     - Memory Analysis
     - Bytecode Details

### Interpreting Results

#### Task Graph View
- Blue nodes represent regular task graphs
- Magenta nodes show Init/End points
- Purple edges indicate data dependencies
- Hover over nodes/edges for detailed information

#### Memory Analysis
- Track memory allocation patterns
- Monitor object lifecycles
- Identify potential memory leaks
- Analyze transfer patterns

## Configuration

The tool can be configured through environment variables:

```bash
export TORNADO_ANALYZER_PORT=8501      # Web interface port
export TORNADO_ANALYZER_HOST=0.0.0.0   # Host address
export TORNADO_ANALYZER_DEBUG=1        # Enable debug mode
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


```
```