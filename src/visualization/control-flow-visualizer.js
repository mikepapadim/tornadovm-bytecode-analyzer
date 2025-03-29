// Control Flow Visualizer React Component
const ControlFlowVisualizer = ({ data }) => {
  const [selectedBlock, setSelectedBlock] = React.useState(null);
  const [layout, setLayout] = React.useState('horizontal');
  const [showCode, setShowCode] = React.useState(true);
  
  // Create a graph using the blocks and edges
  const graph = React.useMemo(() => {
    const nodes = data.blocks.map(block => ({
      id: block.id,
      label: `${block.language.toUpperCase()} Block ${block.id.split('-')[1]}`,
      type: block.language,
      code: block.code,
      lineStart: block.lineStart,
      lineEnd: block.lineEnd
    }));
    
    const edges = data.edges.map(edge => ({
      from: edge.source,
      to: edge.target,
      type: edge.type
    }));
    
    return { nodes, edges };
  }, [data]);
  
  // Handle block selection
  const handleBlockClick = (blockId) => {
    setSelectedBlock(graph.nodes.find(node => node.id === blockId));
  };
  
  return (
    <div className="p-4 bg-gray-950 min-h-screen text-gray-100">
      <header className="mb-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <GitBranch className="text-blue-400" />
          Control Flow Visualization
        </h1>
        <p className="text-gray-400 text-sm">
          Visualize the control flow between code blocks
        </p>
      </header>
      
      {/* Controls */}
      <div className="mb-4 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Layout:</span>
          <div className="flex rounded-md overflow-hidden">
            <button 
              className={`px-3 py-1 text-sm ${layout === 'horizontal' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
              onClick={() => setLayout('horizontal')}
            >
              Horizontal
            </button>
            <button 
              className={`px-3 py-1 text-sm ${layout === 'vertical' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
              onClick={() => setLayout('vertical')}
            >
              Vertical
            </button>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <input 
            type="checkbox" 
            id="show-code" 
            checked={showCode} 
            onChange={(e) => setShowCode(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="show-code" className="text-sm text-gray-400">Show code preview</label>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Graph visualization */}
        <div className="relative h-[600px] border border-gray-700 rounded-lg overflow-hidden">
          <div id="graph-container" className="w-full h-full" />
          <script>
            {`
              // Initialize the graph
              const container = document.getElementById('graph-container');
              const options = {
                nodes: {
                  shape: 'box',
                  margin: 10,
                  font: {
                    size: 14,
                    color: '#ffffff'
                  },
                  borderWidth: 2,
                  shadow: true
                },
                edges: {
                  arrows: 'to',
                  smooth: {
                    type: 'curvedCW',
                    roundness: 0.2
                  },
                  color: {
                    color: '#4B5563',
                    highlight: '#60A5FA'
                  }
                },
                layout: {
                  hierarchical: {
                    direction: '${layout === 'horizontal' ? 'LR' : 'UD'}',
                    sortMethod: 'directed',
                    levelSeparation: 150
                  }
                },
                physics: false
              };
              
              const network = new vis.Network(container, { nodes: new vis.DataSet(${JSON.stringify(graph.nodes)}), edges: new vis.DataSet(${JSON.stringify(graph.edges)}) }, options);
              
              // Handle node selection
              network.on('click', function(params) {
                if (params.nodes.length > 0) {
                  const blockId = params.nodes[0];
                  window.dispatchEvent(new CustomEvent('blockSelected', { detail: blockId }));
                }
              });
            `}
          </script>
        </div>
        
        {/* Code preview */}
        {showCode && selectedBlock && (
          <div className="border border-gray-700 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                {selectedBlock.label}
              </h3>
              <span className="text-sm text-gray-400">
                Lines {selectedBlock.lineStart}-{selectedBlock.lineEnd}
              </span>
            </div>
            <pre className="text-sm overflow-x-auto bg-gray-900 p-4 rounded-lg">
              <code>{selectedBlock.code}</code>
            </pre>
          </div>
        )}
      </div>
      
      <div className="mt-8 border-t border-gray-800 pt-4 text-sm text-gray-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MousePointer size={16} />
            <span>Click on blocks to view their code and explore the control flow</span>
          </div>
          
          <div className="flex items-center gap-2">
            <ExternalLink size={14} />
            <a href="#" className="hover:text-blue-400 transition-colors">View documentation</a>
          </div>
        </div>
      </div>
    </div>
  );
};

// Export the component
window.ControlFlowVisualizer = ControlFlowVisualizer; 