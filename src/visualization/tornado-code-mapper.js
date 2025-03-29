// TornadoVM Code Mapper React Component
const TornadoCodeMapper = ({ data }) => {
  const [selectedJavaBlock, setSelectedJavaBlock] = React.useState(data.java[0]);
  const [selectedTargetType, setSelectedTargetType] = React.useState('ptx');
  const [selectedTargetBlock, setSelectedTargetBlock] = React.useState(data.ptx[0]);
  const [selectedMapping, setSelectedMapping] = React.useState(data.mappings[0]);
  const [selectedJavaLines, setSelectedJavaLines] = React.useState([]);
  const [selectedTargetLines, setSelectedTargetLines] = React.useState([]);
  const [selectedMappings, setSelectedMappings] = React.useState([]);
  const [showTooltip, setShowTooltip] = React.useState(false);
  const [tooltipContent, setTooltipContent] = React.useState(null);
  const [typeFilter, setTypeFilter] = React.useState(null);
  const [autoScroll, setAutoScroll] = React.useState(true);
  
  // Refs for the code containers
  const javaContainerRef = React.useRef(null);
  const targetContainerRef = React.useRef(null);

  // Update selected target when target type changes
  React.useEffect(() => {
    if (selectedTargetType === 'ptx') {
      setSelectedTargetBlock(data.ptx[0]);
      setSelectedMapping(data.mappings[0]);
    } else {
      setSelectedTargetBlock(data.opencl[0]);
      setSelectedMapping(data.mappings[1]);
    }
  }, [selectedTargetType]);

  // Handle Java line click
  const handleJavaLineClick = (lineNumber) => {
    const mapping = selectedMapping;
    let relevantMappings = [];
    
    for (const lineMap of mapping.lineMapping) {
      if (lineMap.sourceLines.includes(lineNumber)) {
        if (!typeFilter || lineMap.type === typeFilter) {
          relevantMappings.push(lineMap);
          
          if (autoScroll && lineMap.targetLines.length > 0 && targetContainerRef.current) {
            const targetLineElement = targetContainerRef.current.querySelector(`[data-line="${lineMap.targetLines[0]}"]`);
            if (targetLineElement) {
              targetLineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }
        }
      }
    }
    
    if (relevantMappings.length > 0) {
      let javaLines = [];
      let targetLines = [];
      
      relevantMappings.forEach(mapping => {
        javaLines = [...javaLines, ...mapping.sourceLines];
        targetLines = [...targetLines, ...mapping.targetLines];
      });
      
      setSelectedJavaLines(javaLines);
      setSelectedTargetLines(targetLines);
      setSelectedMappings(relevantMappings);
      
      setTooltipContent(relevantMappings[0]);
      setShowTooltip(true);
    }
  };
  
  // Handle target line click
  const handleTargetLineClick = (lineNumber) => {
    const mapping = selectedMapping;
    let relevantMappings = [];
    
    for (const lineMap of mapping.lineMapping) {
      if (lineMap.targetLines.includes(lineNumber)) {
        if (!typeFilter || lineMap.type === typeFilter) {
          relevantMappings.push(lineMap);
          
          if (autoScroll && lineMap.sourceLines.length > 0 && javaContainerRef.current) {
            const javaLineElement = javaContainerRef.current.querySelector(`[data-line="${lineMap.sourceLines[0]}"]`);
            if (javaLineElement) {
              javaLineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }
        }
      }
    }
    
    if (relevantMappings.length > 0) {
      let javaLines = [];
      let targetLines = [];
      
      relevantMappings.forEach(mapping => {
        javaLines = [...javaLines, ...mapping.sourceLines];
        targetLines = [...targetLines, ...mapping.targetLines];
      });
      
      setSelectedJavaLines(javaLines);
      setSelectedTargetLines(targetLines);
      setSelectedMappings(relevantMappings);
      
      setTooltipContent(relevantMappings[0]);
      setShowTooltip(true);
    }
  };
  
  // Handle mouse leave for code containers
  const handleMouseLeave = () => {
    setShowTooltip(false);
  };
  
  // Return connected lines DOM positions
  const getConnectionPositions = () => {
    if (!selectedMappings.length || !javaContainerRef.current || !targetContainerRef.current) {
      return [];
    }
    
    const javaContainer = javaContainerRef.current;
    const targetContainer = targetContainerRef.current;
    const javaRect = javaContainer.getBoundingClientRect();
    const targetRect = targetContainer.getBoundingClientRect();
    
    const connections = [];
    
    selectedMappings.forEach(mapping => {
      const sourceLines = mapping.sourceLines;
      const targetLines = mapping.targetLines;
      
      for (let i = 0; i < Math.min(sourceLines.length, 3); i++) {
        const sourceLine = javaContainer.querySelector(`[data-line="${sourceLines[i]}"]`);
        if (!sourceLine) continue;
        
        for (let j = 0; j < Math.min(targetLines.length, 3); j++) {
          const targetLine = targetContainer.querySelector(`[data-line="${targetLines[j]}"]`);
          if (!targetLine) continue;
          
          const sourceRect = sourceLine.getBoundingClientRect();
          const targetRect = targetLine.getBoundingClientRect();
          
          connections.push({
            startY: sourceRect.top + sourceRect.height / 2 - javaRect.top,
            endY: targetRect.top + targetRect.height / 2 - targetRect.top,
            sourceBlockTop: javaContainer.scrollTop,
            targetBlockTop: targetContainer.scrollTop,
            sourceBlockHeight: javaContainer.clientHeight,
            targetBlockHeight: targetContainer.clientHeight,
            color: MAPPING_COLORS[mapping.type] || MAPPING_COLORS.default
          });
        }
      }
    });
    
    return connections;
  };
  
  return (
    <div className="p-4 bg-gray-950 min-h-screen text-gray-100">
      <header className="mb-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Zap className="text-blue-400" />
          TornadoVM Code Transition Visualizer
        </h1>
        <p className="text-gray-400 text-sm">
          Interactive visualization of Java to {selectedTargetType.toUpperCase()} code transformations
        </p>
      </header>
      
      <div className="mb-4 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">Target Language:</span>
            <div className="flex rounded-md overflow-hidden">
              <button 
                className={`px-3 py-1 text-sm ${selectedTargetType === 'ptx' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setSelectedTargetType('ptx')}
              >
                PTX
              </button>
              <button 
                className={`px-3 py-1 text-sm ${selectedTargetType === 'opencl' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setSelectedTargetType('opencl')}
              >
                OpenCL
              </button>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <input 
              type="checkbox" 
              id="auto-scroll" 
              checked={autoScroll} 
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="auto-scroll" className="text-sm text-gray-400">Auto-scroll to mapped code</label>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <GitBranch className="text-gray-400" size={16} />
          <span className="text-sm text-gray-400">Mapping confidence:</span>
          <span className="px-2 py-0.5 bg-green-900 text-green-300 rounded-full text-xs font-medium">
            {(selectedMapping.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        <div 
          ref={javaContainerRef} 
          className="relative" 
          onMouseLeave={handleMouseLeave}
        >
          <CodeBlock
            code={selectedJavaBlock.code}
            language="java"
            lineStart={selectedJavaBlock.lineStart}
            title="Java Source Code"
            selectedLines={selectedJavaLines}
            onLineClick={handleJavaLineClick}
            selectedMappings={selectedMappings}
            targetLines={false}
          />
        </div>
        
        <div className="relative flex">
          <div 
            ref={targetContainerRef} 
            className="flex-grow" 
            onMouseLeave={handleMouseLeave}
          >
            <CodeBlock
              code={selectedTargetBlock.code}
              language={selectedTargetType}
              lineStart={selectedTargetBlock.lineStart}
              title={`${selectedTargetType.toUpperCase()} Generated Code`}
              selectedLines={selectedTargetLines}
              onLineClick={handleTargetLineClick}
              selectedMappings={selectedMappings}
              targetLines={true}
            />
          </div>
          
          {/* Connection arrows */}
          <div className="absolute left-0 top-0 w-full h-full pointer-events-none">
            {getConnectionPositions().map((conn, idx) => (
              <ConnectionArrow key={idx} {...conn} />
            ))}
          </div>
        </div>
      </div>
      
      {/* Mapping details */}
      {tooltipContent && showTooltip && (
        <div className="mt-4">
          {renderMappingDetails(tooltipContent)}
        </div>
      )}
      
      {/* Mapping legend */}
      <MappingLegend activeFilter={typeFilter} setActiveFilter={setTypeFilter} />
      
      <div className="mt-8 border-t border-gray-800 pt-4 text-sm text-gray-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen size={16} />
            <span>Click on code lines to see their mappings between Java and {selectedTargetType.toUpperCase()}</span>
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
window.TornadoCodeMapper = TornadoCodeMapper; 