import sys
import os
import numpy as np

# Add parent directory to path to import the visualizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tornado_code_transition import TornadoCodeVisualizer

def test_dft_visualization():
    # Initialize the visualizer
    visualizer = TornadoCodeVisualizer()
    
    # Load the source files
    java_source = os.path.join(os.path.dirname(__file__), 'DFT.java')
    opencl_source = os.path.join(os.path.dirname(__file__), 'dft.opencl')
    ptx_source = os.path.join(os.path.dirname(__file__), 'dft.ptx')
    
    # Add the code mappings
    visualizer.add_code_mapping(
        source_file=java_source,
        target_files=[opencl_source, ptx_source],
        method_name='computeDFTFloat'
    )
    
    # Generate test data
    size = 4096
    input_real = np.random.rand(size).astype(np.float32)
    input_imag = np.random.rand(size).astype(np.float32)
    
    # Add performance metrics (simulated for this example)
    metrics = {
        'execution_time': 0.0023,  # seconds
        'memory_transfers': {
            'host_to_device': 32768,  # bytes
            'device_to_host': 32768   # bytes
        },
        'occupancy': 0.85,
        'thread_blocks': 256,
        'threads_per_block': 256
    }
    visualizer.add_performance_metrics(metrics)
    
    # Generate and display the visualization
    visualizer.generate_visualization()
    
if __name__ == '__main__':
    test_dft_visualization() 