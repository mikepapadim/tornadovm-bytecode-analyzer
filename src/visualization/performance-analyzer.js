// Performance Analyzer React Component
const PerformanceAnalyzer = ({ data }) => {
  const [selectedLanguage, setSelectedLanguage] = React.useState(Object.keys(data)[0]);
  const [selectedMetric, setSelectedMetric] = React.useState('instruction_mix');
  const [selectedBlock, setSelectedBlock] = React.useState(null);
  
  // Get available metrics from the first block
  const availableMetrics = React.useMemo(() => {
    if (!data[selectedLanguage]?.length) return [];
    const firstBlock = data[selectedLanguage][0];
    return Object.keys(firstBlock.metrics || {});
  }, [data, selectedLanguage]);
  
  // Calculate aggregated metrics
  const aggregatedMetrics = React.useMemo(() => {
    if (!data[selectedLanguage]?.length) return null;
    
    const blocks = data[selectedLanguage];
    const metrics = {};
    
    blocks.forEach(block => {
      if (!block.metrics) return;
      
      Object.entries(block.metrics).forEach(([key, value]) => {
        if (!metrics[key]) {
          metrics[key] = {};
        }
        
        if (typeof value === 'number') {
          metrics[key].total = (metrics[key].total || 0) + value;
          metrics[key].average = metrics[key].total / blocks.length;
        } else if (Array.isArray(value)) {
          metrics[key].total = (metrics[key].total || 0) + value.length;
          metrics[key].average = metrics[key].total / blocks.length;
        } else if (typeof value === 'object') {
          Object.entries(value).forEach(([subKey, subValue]) => {
            if (!metrics[key][subKey]) {
              metrics[key][subKey] = {};
            }
            
            if (typeof subValue === 'number') {
              metrics[key][subKey].total = (metrics[key][subKey].total || 0) + subValue;
              metrics[key][subKey].average = metrics[key][subKey].total / blocks.length;
            } else if (Array.isArray(subValue)) {
              metrics[key][subKey].total = (metrics[key][subKey].total || 0) + subValue.length;
              metrics[key][subKey].average = metrics[key][subKey].total / blocks.length;
            }
          });
        }
      });
    });
    
    return metrics;
  }, [data, selectedLanguage]);
  
  return (
    <div className="p-4 bg-gray-950 min-h-screen text-gray-100">
      <header className="mb-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Zap className="text-blue-400" />
          Performance Analysis
        </h1>
        <p className="text-gray-400 text-sm">
          Analyze performance metrics across different code blocks
        </p>
      </header>
      
      {/* Language selector */}
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Language:</span>
          <div className="flex rounded-md overflow-hidden">
            {Object.keys(data).map(lang => (
              <button 
                key={lang}
                className={`px-3 py-1 text-sm ${selectedLanguage === lang ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setSelectedLanguage(lang)}
              >
                {lang.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>
      
      {/* Metric selector */}
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Metric:</span>
          <div className="flex rounded-md overflow-hidden">
            {availableMetrics.map(metric => (
              <button 
                key={metric}
                className={`px-3 py-1 text-sm ${selectedMetric === metric ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setSelectedMetric(metric)}
              >
                {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </button>
            ))}
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Code blocks */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Code Blocks</h2>
          <div className="space-y-2">
            {data[selectedLanguage]?.map((block, index) => (
              <div 
                key={block.id}
                className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                  selectedBlock?.id === block.id 
                    ? 'border-blue-500 bg-gray-800' 
                    : 'border-gray-700 hover:border-gray-600'
                }`}
                onClick={() => setSelectedBlock(block)}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Block {index + 1}</span>
                  <span className="text-xs text-gray-400">
                    Lines {block.lineStart}-{block.lineEnd}
                  </span>
                </div>
                <pre className="text-xs overflow-x-auto">
                  <code>{block.code}</code>
                </pre>
              </div>
            ))}
          </div>
        </div>
        
        {/* Metrics visualization */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Performance Metrics</h2>
          {selectedBlock && selectedBlock.metrics && (
            <div className="space-y-4">
              {/* Block-specific metrics */}
              <div className="p-4 rounded-lg border border-gray-700">
                <h3 className="text-sm font-medium mb-2">Block Metrics</h3>
                <div className="space-y-2">
                  {Object.entries(selectedBlock.metrics[selectedMetric] || {}).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className="text-sm font-medium">
                        {typeof value === 'number' ? value : value.length}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Aggregated metrics */}
              {aggregatedMetrics && (
                <div className="p-4 rounded-lg border border-gray-700">
                  <h3 className="text-sm font-medium mb-2">Aggregated Metrics</h3>
                  <div className="space-y-2">
                    {Object.entries(aggregatedMetrics[selectedMetric] || {}).map(([key, value]) => (
                      <div key={key} className="flex justify-between items-center">
                        <span className="text-sm text-gray-400">
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <span className="text-sm font-medium">
                          {typeof value === 'object' ? value.average.toFixed(2) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Export the component
window.PerformanceAnalyzer = PerformanceAnalyzer; 