// NOTEARS Web GUI JavaScript
const socket = io();
let currentDataset = null;
let currentResults = null;
let uploadedCSVData = null;
let uploadedBIFContent = null;
let editingDataset = null;
let availableAlgorithms = null;
let currentAlgorithm = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableDatasets();
    loadAvailableAlgorithms();
    setupSocketHandlers();
    updateStatus('Ready', 'ready');
});

// Socket.IO event handlers
function setupSocketHandlers() {
    socket.on('log_message', function(data) {
        addLogMessage(data.message, data.level);
    });
    
    socket.on('algorithm_progress', function(data) {
        updateProgress(data.iteration, data.metric_value, data.additional_info);
    });
    
    socket.on('algorithm_completed', function(data) {
        handleAlgorithmCompletion(data);
    });
    
    socket.on('algorithm_failed', function(data) {
        handleAlgorithmFailure(data.error);
    });
    
    socket.on('status_update', function(data) {
        updateStatus(data.message, data.status);
    });
}

// Load available datasets
async function loadAvailableDatasets() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        
        const select = document.getElementById('datasetSelect');
        select.innerHTML = '<option value="">Select a dataset...</option>';
        
        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.name;
            option.textContent = `${dataset.name} (${dataset.nodes} nodes, ${dataset.edges} edges)`;
            select.appendChild(option);
        });
        
        addLogMessage(`Loaded ${datasets.length} available datasets`, 'info');
    } catch (error) {
        addLogMessage(`Error loading datasets: ${error.message}`, 'error');
    }
}

// Load available algorithms
async function loadAvailableAlgorithms() {
    try {
        const response = await fetch('/api/algorithms');
        availableAlgorithms = await response.json();
        
        const select = document.getElementById('algorithmSelect');
        select.innerHTML = '<option value="">Select an algorithm...</option>';
        
        availableAlgorithms.forEach(algorithm => {
            const option = document.createElement('option');
            option.value = algorithm.id;
            option.textContent = algorithm.name;
            select.appendChild(option);
        });
        
        // Auto-select the first algorithm (typically NOTEARS Nonlinear)
        if (availableAlgorithms.length > 0) {
            select.value = availableAlgorithms[0].id;
            loadAlgorithmParameters();
        }
        
        addLogMessage(`Loaded ${availableAlgorithms.length} available algorithms`, 'info');
    } catch (error) {
        addLogMessage(`Error loading algorithms: ${error.message}`, 'error');
    }
}

// Load algorithm parameters based on selection
function loadAlgorithmParameters() {
    const select = document.getElementById('algorithmSelect');
    const algorithmId = select.value;
    const description = document.getElementById('algorithmDescription');
    const container = document.getElementById('parametersContainer');
    
    if (!algorithmId) {
        description.textContent = 'Select an algorithm to see its description';
        container.innerHTML = `
            <div class="text-center text-muted py-3">
                <i class="fas fa-cogs fa-2x mb-2"></i>
                <p>Select an algorithm to configure parameters</p>
            </div>
        `;
        currentAlgorithm = null;
        return;
    }
    
    // Find algorithm info
    currentAlgorithm = availableAlgorithms.find(alg => alg.id === algorithmId);
    if (!currentAlgorithm) {
        showAlert('Algorithm not found', 'danger');
        return;
    }
    
    // Update description
    description.textContent = currentAlgorithm.description;
    
    // Generate parameter controls
    let parametersHtml = '';
    
    Object.entries(currentAlgorithm.parameters).forEach(([paramName, paramDef]) => {
        const inputId = `param_${paramName}`;
        
        parametersHtml += '<div class="mb-3">';
        parametersHtml += `<label for="${inputId}" class="form-label">${formatParameterName(paramName)}:</label>`;
        
        if (paramDef.type === 'choice') {
            parametersHtml += `<select class="form-control" id="${inputId}">`;
            paramDef.choices.forEach(choice => {
                const selected = choice === paramDef.default ? 'selected' : '';
                parametersHtml += `<option value="${choice}" ${selected}>${choice}</option>`;
            });
            parametersHtml += '</select>';
        } else if (paramDef.type === 'bool') {
            const checked = paramDef.default ? 'checked' : '';
            parametersHtml += `<div class="form-check">`;
            parametersHtml += `<input class="form-check-input" type="checkbox" id="${inputId}" ${checked}>`;
            parametersHtml += `<label class="form-check-label" for="${inputId}">Enable</label>`;
            parametersHtml += `</div>`;
        } else {
            // Numeric or text input
            const inputType = paramDef.type === 'int' || paramDef.type === 'float' ? 'number' : 'text';
            let inputAttrs = `type="${inputType}" class="form-control" id="${inputId}" value="${paramDef.default}"`;
            
            if (paramDef.type === 'int' || paramDef.type === 'float') {
                if (paramDef.min !== undefined) inputAttrs += ` min="${paramDef.min}"`;
                if (paramDef.max !== undefined) inputAttrs += ` max="${paramDef.max}"`;
                if (paramDef.step !== undefined) inputAttrs += ` step="${paramDef.step}"`;
            }
            
            parametersHtml += `<input ${inputAttrs}>`;
        }
        
        parametersHtml += `<small class="form-text text-muted">${paramDef.description}</small>`;
        parametersHtml += '</div>';
    });
    
    container.innerHTML = parametersHtml;
    addLogMessage(`Loaded parameters for ${currentAlgorithm.name}`, 'info');
}

// Format parameter name for display
function formatParameterName(paramName) {
    return paramName
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
}

// Load selected dataset
async function loadSelectedDataset() {
    const select = document.getElementById('datasetSelect');
    const datasetName = select.value;
    
    if (!datasetName) {
        showAlert('Please select a dataset first.', 'warning');
        return;
    }
    
    try {
        updateStatus('Loading dataset...', 'running');
        const response = await fetch(`/api/load_dataset/${datasetName}`);
        const result = await response.json();
        
        if (result.success) {
            currentDataset = result;
            displayDatasetInfo(result);
            displayDataPreview(result.preview, result.columns);
            updateStatus(`Loaded: ${result.dataset_name}`, 'ready');
            addLogMessage(`Successfully loaded dataset: ${result.dataset_name}`, 'success');
        } else {
            showAlert(`Error loading dataset: ${result.error}`, 'danger');
            updateStatus('Error loading dataset', 'error');
        }
    } catch (error) {
        showAlert(`Error loading dataset: ${error.message}`, 'danger');
        updateStatus('Error', 'error');
        addLogMessage(`Error loading dataset: ${error.message}`, 'error');
    }
}

// Display dataset information
function displayDatasetInfo(dataset) {
    const infoDiv = document.getElementById('datasetInfo');
    const detailsDiv = document.getElementById('datasetDetails');
    
    let infoHtml = `
        <div class="row">
            <div class="col-6"><strong>Name:</strong></div>
            <div class="col-6">${dataset.dataset_name}</div>
        </div>
        <div class="row">
            <div class="col-6"><strong>Shape:</strong></div>
            <div class="col-6">${dataset.shape[0]} × ${dataset.shape[1]}</div>
        </div>
        <div class="row">
            <div class="col-6"><strong>Variables:</strong></div>
            <div class="col-6">${dataset.columns.join(', ')}</div>
        </div>
    `;
    
    if (dataset.info && dataset.info.description) {
        infoHtml += `
            <div class="row">
                <div class="col-6"><strong>Description:</strong></div>
                <div class="col-6">${dataset.info.description}</div>
            </div>
        `;
    }
    
    if (dataset.has_ground_truth) {
        infoHtml += `
            <div class="row">
                <div class="col-12">
                    <span class="badge bg-success">
                        <i class="fas fa-check me-1"></i>Ground truth available
                    </span>
                </div>
            </div>
        `;
    }
    
    detailsDiv.innerHTML = infoHtml;
    infoDiv.style.display = 'block';
}

// Display data preview
function displayDataPreview(data, columns) {
    const previewDiv = document.getElementById('dataPreview');
    
    if (!data || data.length === 0) {
        previewDiv.innerHTML = '<div class="text-center text-muted py-3">No data to preview</div>';
        return;
    }
    
    let tableHtml = `
        <div class="table-responsive">
            <table class="table table-sm dataset-table">
                <thead>
                    <tr>
                        ${columns.map(col => `<th>${col}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;
    
    data.forEach(row => {
        tableHtml += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
            tableHtml += `<td>${displayValue}</td>`;
        });
        tableHtml += '</tr>';
    });
    
    tableHtml += '</tbody></table></div>';
    
    if (data.length >= 10) {
        tableHtml += '<div class="text-muted text-center mt-2"><small>Showing first 10 rows</small></div>';
    }
    
    previewDiv.innerHTML = tableHtml;
}

// Run NOTEARS algorithm
function runAlgorithm() {
    if (!currentDataset) {
        showAlert('Please load a dataset first.', 'warning');
        return;
    }
    
    if (!currentAlgorithm) {
        showAlert('Please select an algorithm first.', 'warning');
        return;
    }
    
    // Collect parameters from the form
    const params = {};
    Object.keys(currentAlgorithm.parameters).forEach(paramName => {
        const element = document.getElementById(`param_${paramName}`);
        if (element) {
            const paramDef = currentAlgorithm.parameters[paramName];
            
            if (paramDef.type === 'bool') {
                params[paramName] = element.checked;
            } else if (paramDef.type === 'int') {
                params[paramName] = parseInt(element.value);
            } else if (paramDef.type === 'float') {
                params[paramName] = parseFloat(element.value);
            } else {
                params[paramName] = element.value;
            }
        }
    });
    
    // Basic validation
    let validationFailed = false;
    Object.entries(currentAlgorithm.parameters).forEach(([paramName, paramDef]) => {
        const value = params[paramName];
        
        if (paramDef.type === 'int' || paramDef.type === 'float') {
            if (isNaN(value) || 
                (paramDef.min !== undefined && value < paramDef.min) ||
                (paramDef.max !== undefined && value > paramDef.max)) {
                validationFailed = true;
            }
        }
    });
    
    if (validationFailed) {
        showAlert('Please check parameter values.', 'warning');
        return;
    }
    
    // Update UI
    document.getElementById('runButton').style.display = 'none';
    document.getElementById('stopButton').style.display = 'block';
    document.getElementById('progressContainer').style.display = 'block';
    updateStatus('Running NOTEARS algorithm...', 'running');
    
    // Clear previous results
    currentResults = null;
    document.getElementById('resultsSummary').innerHTML = `
        <div class="text-center">
            <div class="spinner"></div>
            <p>Running algorithm...</p>
        </div>
    `;
    
    // Send request to server
    socket.emit('run_algorithm', {
        dataset: currentDataset.dataset_name,
        algorithm: currentAlgorithm.id,
        parameters: params
    });
    
    addLogMessage(`Starting ${currentAlgorithm.name}...`, 'info');
    const paramStr = Object.entries(params).map(([k, v]) => `${k}=${v}`).join(', ');
    addLogMessage(`Parameters: ${paramStr}`, 'info');
}

// Stop algorithm
function stopAlgorithm() {
    socket.emit('stop_algorithm');
    resetAlgorithmUI();
    updateStatus('Algorithm stopped', 'ready');
    addLogMessage('Algorithm stopped by user', 'warning');
}

// Handle algorithm completion
function handleAlgorithmCompletion(data) {
    currentResults = data;
    resetAlgorithmUI();
    updateStatus('Algorithm completed successfully', 'ready');
    
    displayResults(data);
    
    // Switch to results tab
    const resultsTab = new bootstrap.Tab(document.getElementById('results-tab'));
    resultsTab.show();
    
    addLogMessage(`Algorithm completed in ${data.runtime.toFixed(2)} seconds`, 'success');
}

// Handle algorithm failure
function handleAlgorithmFailure(error) {
    resetAlgorithmUI();
    updateStatus('Algorithm failed', 'error');
    showAlert(`Algorithm failed: ${error}`, 'danger');
    addLogMessage(`Algorithm failed: ${error}`, 'error');
}

// Reset algorithm UI
function resetAlgorithmUI() {
    document.getElementById('runButton').style.display = 'block';
    document.getElementById('stopButton').style.display = 'none';
    document.getElementById('progressContainer').style.display = 'none';
}

// Update algorithm progress
function updateProgress(iteration, metricValue, additionalInfo) {
    if (additionalInfo !== undefined && additionalInfo !== null) {
        // NOTEARS format: h value and rho
        addLogMessage(`Iter ${iteration}: h=${metricValue.toFixed(8)}, ρ=${additionalInfo.toExponential(2)}`, 'info');
    } else {
        // Generic format
        addLogMessage(`Iter ${iteration}: metric=${metricValue.toFixed(6)}`, 'info');
    }
}

// Display results
function displayResults(results) {
    const summaryDiv = document.getElementById('resultsSummary');
    
    let summaryHtml = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${results.runtime.toFixed(2)}s</div>
                    <div class="metric-label">Runtime</div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${results.learned_edges}</div>
                    <div class="metric-label">Learned Edges</div>
                </div>
            </div>
        </div>
    `;
    
    if (results.metrics) {
        summaryHtml += `
            <div class="row mt-3">
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${results.metrics.precision.toFixed(3)}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${results.metrics.recall.toFixed(3)}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${results.metrics.f1_score.toFixed(3)}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    summaryDiv.innerHTML = summaryHtml;
    
    // Enable comparison view if ground truth is available
    const compareViewRadio = document.getElementById('compareView');
    if (results.ground_truth_matrix) {
        compareViewRadio.disabled = false;
        compareViewRadio.parentElement.title = 'Compare learned graph with ground truth';
    } else {
        compareViewRadio.disabled = true;
        compareViewRadio.parentElement.title = 'Ground truth not available for comparison';
        // Reset to single view if compare was selected
        if (currentViewMode === 'compare') {
            document.getElementById('singleView').checked = true;
            setViewMode('single');
        }
    }
    
    // Enable save results button
    document.getElementById('saveResultsBtn').disabled = false;
    
    // Show default visualization
    updateVisualization();
}

// Global visualization state
let currentViewMode = 'single';
let currentVisualizationType = 'graph';

// Set view mode (single or compare)
function setViewMode(mode) {
    currentViewMode = mode;
    
    // Update UI
    if (mode === 'single') {
        document.getElementById('singleViewContainer').style.display = 'block';
        document.getElementById('compareViewContainer').style.display = 'none';
    } else {
        document.getElementById('singleViewContainer').style.display = 'none';
        document.getElementById('compareViewContainer').style.display = 'block';
    }
    
    // Update visualization
    updateVisualization();
}

// Set visualization type (graph or heatmap)
function setVisualizationType(type) {
    currentVisualizationType = type;
    updateVisualization();
}

// Update visualization based on current settings
function updateVisualization() {
    if (!currentResults) {
        return;
    }
    
    
    if (currentViewMode === 'single') {
        const container = document.getElementById('singleVisualization');
        if (currentVisualizationType === 'graph') {
            createGraphVisualization(container, currentResults, 'Learned Causal Graph');
        } else {
            createHeatmapVisualization(container, currentResults, 'Learned Adjacency Matrix');
        }
    } else {
        // Compare mode
        const learnedContainer = document.getElementById('learnedVisualization');
        const groundTruthContainer = document.getElementById('groundTruthVisualization');
        
        if (currentVisualizationType === 'graph') {
            createGraphVisualization(learnedContainer, currentResults, 'Learned Graph', 'learned');
            createGroundTruthGraphVisualization(groundTruthContainer, currentResults);
        } else {
            createHeatmapVisualization(learnedContainer, currentResults, 'Learned Matrix', 'learned');
            createGroundTruthHeatmapVisualization(groundTruthContainer, currentResults);
        }
        
        // Update comparison statistics
        updateComparisonStatistics();
    }
}

// Create graph visualization using Plotly
function createGraphVisualization(container, results, title = 'Causal Graph', type = 'single') {
    const adjMatrix = results.adjacency_matrix;
    const nodeNames = results.node_names;
    const threshold = results.parameters.threshold;
    
    // Create nodes in a circle layout
    const nodes = nodeNames.map((name, i) => ({
        id: i,
        label: name,
        x: Math.cos(2 * Math.PI * i / nodeNames.length),
        y: Math.sin(2 * Math.PI * i / nodeNames.length)
    }));
    
    // Extract edges
    const edges = [];
    for (let i = 0; i < adjMatrix.length; i++) {
        for (let j = 0; j < adjMatrix[i].length; j++) {
            if (adjMatrix[i][j] > threshold) {
                edges.push({
                    source: i,
                    target: j,
                    weight: adjMatrix[i][j]
                });
            }
        }
    }
    
    // Create edge trace (no hover to avoid interference)
    const edgeTrace = {
        x: [],
        y: [],
        mode: 'lines',
        line: { 
            width: 2, 
            color: type === 'learned' ? '#2E86AB' : '#A23B72' 
        },
        hoverinfo: 'skip',  // Skip hover for edges
        showlegend: false
    };
    
    // Create arrow shapes for directed edges
    const shapes = [];
    edges.forEach(edge => {
        const sourceNode = nodes[edge.source];
        const targetNode = nodes[edge.target];
        
        // Calculate arrow position (closer to target node)
        const dx = targetNode.x - sourceNode.x;
        const dy = targetNode.y - sourceNode.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const arrowX = targetNode.x - (dx / length) * 0.15;
        const arrowY = targetNode.y - (dy / length) * 0.15;
        
        // Add line to edge trace
        edgeTrace.x.push(sourceNode.x, arrowX, null);
        edgeTrace.y.push(sourceNode.y, arrowY, null);
        
        // Arrow head
        const angle = Math.atan2(dy, dx);
        const arrowSize = 0.05;
        shapes.push({
            type: 'path',
            path: `M ${arrowX},${arrowY} L ${arrowX - arrowSize * Math.cos(angle - Math.PI/6)},${arrowY - arrowSize * Math.sin(angle - Math.PI/6)} L ${arrowX - arrowSize * Math.cos(angle + Math.PI/6)},${arrowY - arrowSize * Math.sin(angle + Math.PI/6)} Z`,
            fillcolor: type === 'learned' ? '#2E86AB' : '#A23B72',
            line: { width: 0 }
        });
    });
    
    // Calculate edge counts for each node
    const nodeEdgeCounts = nodes.map((node, nodeIndex) => {
        const incomingEdges = edges.filter(e => e.target === nodeIndex).length;
        const outgoingEdges = edges.filter(e => e.source === nodeIndex).length;
        return { incoming: incomingEdges, outgoing: outgoingEdges, total: incomingEdges + outgoingEdges };
    });

    // Create separate marker and text traces for better hover detection
    const markerTrace = {
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        mode: 'markers',
        marker: {
            size: 40,  // Larger for better hover detection
            color: type === 'learned' ? '#2E86AB' : '#A23B72',
            line: { width: 2, color: 'white' },
            opacity: 0.9
        },
        customdata: nodeEdgeCounts.map((count, i) => ({ ...count, name: nodes[i].label })),
        hovertemplate: '<b>%{customdata.name}</b><br>' +
                      'Incoming: %{customdata.incoming}<br>' +
                      'Outgoing: %{customdata.outgoing}<br>' +
                      'Total: %{customdata.total}' +
                      '<extra></extra>',
        hoverlabel: {
            bgcolor: 'white',
            bordercolor: type === 'learned' ? '#2E86AB' : '#A23B72',
            font: { color: 'black', size: 12 }
        },
        showlegend: false
    };
    
    // Create text trace (no hover)
    const textTrace = {
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        mode: 'text',
        text: nodes.map(n => n.label),
        textposition: 'middle center',
        textfont: { size: 12, color: 'white', family: 'Arial Black' },
        hoverinfo: 'skip',  // Skip hover for text
        showlegend: false
    };
    
    const layout = {
        title: {
            text: title,
            x: 0.5,
            font: { size: 14 }
        },
        showlegend: false,
        xaxis: { 
            visible: false, 
            range: [-1.5, 1.5],
            fixedrange: true
        },
        yaxis: { 
            visible: false, 
            range: [-1.5, 1.5],
            fixedrange: true
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 20, r: 20, t: 40, b: 20 },
        shapes: shapes,
        hovermode: 'closest',  // Ensure hover works on closest point
        hoverdistance: 100,    // Increase hover detection distance
        spikedistance: 100,    // Additional hover detection improvement
        annotations: [{
            text: `${edges.length} edges (threshold=${threshold})`,
            showarrow: false,
            x: 0.5,
            y: -0.1,
            xref: 'paper',
            yref: 'paper',
            font: { size: 10, color: 'gray' }
        }]
    };
    
    const config = { 
        responsive: true, 
        displayModeBar: false,
        scrollZoom: false,
        doubleClick: false,
        showTips: false,
        dragmode: false,
        staticPlot: false  // Allow hover but no other interactions
    };
    
    // Clear any existing content (including placeholder text)
    container.innerHTML = '';
    
    Plotly.newPlot(container, [edgeTrace, markerTrace, textTrace], layout, config);
}

// Create ground truth graph visualization
function createGroundTruthGraphVisualization(container, results) {
    if (!results.ground_truth_matrix) {
        container.innerHTML = `
            <div class="text-center text-muted py-5">
                <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                <p>No ground truth available</p>
                <small>Upload a dataset with BIF file to see ground truth comparison</small>
            </div>
        `;
        return;
    }
    
    // Create a mock results object for ground truth
    const groundTruthResults = {
        adjacency_matrix: results.ground_truth_matrix,
        node_names: results.node_names,
        parameters: { threshold: 0.5 } // Ground truth is binary
    };
    
    createGraphVisualization(container, groundTruthResults, 'Ground Truth Graph', 'ground_truth');
}

// Create heatmap visualization
function createHeatmapVisualization(container, results, title = 'Adjacency Matrix', type = 'single') {
    const adjMatrix = results.adjacency_matrix;
    const nodeNames = results.node_names;
    
    const data = [{
        z: adjMatrix,
        x: nodeNames,
        y: nodeNames,
        type: 'heatmap',
        colorscale: type === 'learned' ? 'Blues' : 'Reds',
        showscale: true,
        hoverongaps: false,
        hovertemplate: '%{y} → %{x}<br>Weight: %{z:.4f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: title,
            x: 0.5,
            font: { size: 14 }
        },
        xaxis: { 
            title: 'To Node',
            tickangle: -45,
            fixedrange: true
        },
        yaxis: { 
            title: 'From Node',
            autorange: 'reversed',
            fixedrange: true
        },
        margin: { l: 80, r: 50, t: 50, b: 80 },
        plot_bgcolor: 'white'
    };
    
    const config = { 
        responsive: true, 
        displayModeBar: false,
        scrollZoom: false,
        doubleClick: false,
        showTips: false,
        dragmode: false,
        staticPlot: false  // Allow hover but no other interactions
    };
    
    // Clear any existing content (including placeholder text)
    container.innerHTML = '';
    
    Plotly.newPlot(container, data, layout, config);
}

// Create ground truth heatmap visualization
function createGroundTruthHeatmapVisualization(container, results) {
    if (!results.ground_truth_matrix) {
        container.innerHTML = `
            <div class="text-center text-muted py-5">
                <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                <p>No ground truth available</p>
                <small>Upload a dataset with BIF file to see ground truth comparison</small>
            </div>
        `;
        return;
    }
    
    const groundTruthResults = {
        adjacency_matrix: results.ground_truth_matrix,
        node_names: results.node_names,
        parameters: { threshold: 0.5 }
    };
    
    createHeatmapVisualization(container, groundTruthResults, 'Ground Truth Matrix', 'ground_truth');
}

// Update comparison statistics
function updateComparisonStatistics() {
    if (!currentResults || !currentResults.ground_truth_matrix) {
        document.getElementById('comparisonStats').style.display = 'none';
        return;
    }
    
    const learned = currentResults.adjacency_matrix;
    const groundTruth = currentResults.ground_truth_matrix;
    const threshold = currentResults.parameters.threshold;
    
    // Convert to binary matrices
    const learnedBinary = learned.map(row => row.map(val => val > threshold ? 1 : 0));
    const groundTruthBinary = groundTruth.map(row => row.map(val => val > 0 ? 1 : 0));
    
    // Calculate edge statistics
    let correctEdges = 0;
    let missedEdges = 0;
    let extraEdges = 0;
    let totalGroundTruthEdges = 0;
    let totalLearnedEdges = 0;
    
    for (let i = 0; i < learned.length; i++) {
        for (let j = 0; j < learned[i].length; j++) {
            if (i !== j) { // Exclude diagonal
                const learnedEdge = learnedBinary[i][j];
                const trueEdge = groundTruthBinary[i][j];
                
                if (trueEdge === 1) {
                    totalGroundTruthEdges++;
                    if (learnedEdge === 1) {
                        correctEdges++;
                    } else {
                        missedEdges++;
                    }
                } else if (learnedEdge === 1) {
                    extraEdges++;
                }
                
                if (learnedEdge === 1) {
                    totalLearnedEdges++;
                }
            }
        }
    }
    
    const edgeAccuracy = totalGroundTruthEdges > 0 ? (correctEdges / totalGroundTruthEdges * 100).toFixed(1) + '%' : 'N/A';
    
    // Update UI
    document.getElementById('correctEdges').textContent = correctEdges;
    document.getElementById('missedEdges').textContent = missedEdges;
    document.getElementById('extraEdges').textContent = extraEdges;
    document.getElementById('edgeAccuracy').textContent = edgeAccuracy;
    
    document.getElementById('comparisonStats').style.display = 'block';
}

// Export results
function exportResults(format) {
    if (!currentResults) {
        showAlert('No results to export. Run the algorithm first.', 'warning');
        return;
    }
    
    if (format === 'png') {
        // Handle PNG export through visualization
        exportVisualizationAsPNG();
    } else {
        // Handle CSV and JSON export through backend API
        const params = new URLSearchParams({
            format: format,
            dataset: currentDataset.dataset_name
        });
        
        window.open(`/api/export_results?${params.toString()}`, '_blank');
        addLogMessage(`Exported results in ${format.toUpperCase()} format`, 'info');
    }
}

// Export visualization as PNG
function exportVisualizationAsPNG() {
    if (!currentResults) {
        showAlert('No results to export. Run the algorithm first.', 'warning');
        return;
    }
    
    // Show export options modal
    showExportOptionsModal();
}

// Show export options modal
function showExportOptionsModal() {
    // Create modal HTML if it doesn't exist
    let modal = document.getElementById('exportOptionsModal');
    if (!modal) {
        createExportOptionsModal();
        modal = document.getElementById('exportOptionsModal');
    }
    
    // Update options based on current state
    updateExportOptions();
    
    // Show modal
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
}

// Create export options modal
function createExportOptionsModal() {
    const modalHTML = `
        <div class="modal fade" id="exportOptionsModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-download me-2"></i>Export Visualization
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Export Type:</label>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="exportType" id="exportSingle" value="single" checked>
                                <label class="form-check-label" for="exportSingle">
                                    Current View Only
                                </label>
                            </div>
                            <div class="form-check" id="exportLearnedOption">
                                <input class="form-check-input" type="radio" name="exportType" id="exportLearned" value="learned">
                                <label class="form-check-label" for="exportLearned">
                                    Learned Graph Only
                                </label>
                            </div>
                            <div class="form-check" id="exportGroundTruthOption">
                                <input class="form-check-input" type="radio" name="exportType" id="exportGroundTruth" value="groundtruth">
                                <label class="form-check-label" for="exportGroundTruth">
                                    Ground Truth Only
                                </label>
                            </div>
                            <div class="form-check" id="exportSideBySideOption">
                                <input class="form-check-input" type="radio" name="exportType" id="exportSideBySide" value="sidebyside">
                                <label class="form-check-label" for="exportSideBySide">
                                    Side-by-Side Comparison
                                </label>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Visualization Type:</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="exportVizType" id="exportGraph" value="graph" checked>
                                <label class="btn btn-outline-secondary" for="exportGraph">
                                    <i class="fas fa-share-alt me-1"></i>Graph
                                </label>
                                <input type="radio" class="btn-check" name="exportVizType" id="exportHeatmap" value="heatmap">
                                <label class="btn btn-outline-secondary" for="exportHeatmap">
                                    <i class="fas fa-th me-1"></i>Heatmap
                                </label>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="exportResolution" class="form-label">Resolution:</label>
                            <select class="form-select" id="exportResolution">
                                <option value="800x600">Standard (800x600)</option>
                                <option value="1200x900" selected>High (1200x900)</option>
                                <option value="1600x1200">Very High (1600x1200)</option>
                                <option value="2400x1800">Ultra High (2400x1800)</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="exportFilename" class="form-label">Filename:</label>
                            <input type="text" class="form-control" id="exportFilename" placeholder="Custom filename (optional)">
                            <small class="form-text text-muted">Leave empty for auto-generated filename</small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="performExport()">
                            <i class="fas fa-download me-1"></i>Export PNG
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Update export options based on current state
function updateExportOptions() {
    const hasGroundTruth = currentResults && currentResults.ground_truth_matrix;
    
    // Show/hide ground truth options
    document.getElementById('exportGroundTruthOption').style.display = hasGroundTruth ? 'block' : 'none';
    document.getElementById('exportSideBySideOption').style.display = hasGroundTruth ? 'block' : 'none';
    
    // Update current view description
    const currentViewLabel = document.querySelector('label[for="exportSingle"]');
    if (currentViewMode === 'single') {
        currentViewLabel.textContent = `Current View (${currentVisualizationType} - learned graph)`;
    } else {
        currentViewLabel.textContent = `Current View (${currentVisualizationType} - comparison mode)`;
    }
    
    // Set default visualization type to current
    document.getElementById('exportGraph').checked = currentVisualizationType === 'graph';
    document.getElementById('exportHeatmap').checked = currentVisualizationType === 'heatmap';
}

// Perform the actual export
async function performExport() {
    const exportType = document.querySelector('input[name="exportType"]:checked').value;
    const vizType = document.querySelector('input[name="exportVizType"]:checked').value;
    const resolution = document.getElementById('exportResolution').value;
    const customFilename = document.getElementById('exportFilename').value.trim();
    
    const [width, height] = resolution.split('x').map(Number);
    
    try {
        let filename = customFilename || generateExportFilename(exportType, vizType);
        if (!filename.endsWith('.png')) {
            filename += '.png';
        }
        
        if (exportType === 'sidebyside') {
            await exportSideBySideComparison(vizType, width, height, filename);
        } else {
            await exportSingleVisualization(exportType, vizType, width, height, filename);
        }
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('exportOptionsModal'));
        modal.hide();
        
    } catch (error) {
        console.error('Export error:', error);
        showAlert('Failed to export visualization.', 'danger');
    }
}

// Export single visualization
async function exportSingleVisualization(exportType, vizType, width, height, filename) {
    let targetContainer;
    
    if (exportType === 'single') {
        // Export current view
        if (currentViewMode === 'single') {
            targetContainer = document.getElementById('singleVisualization');
        } else {
            targetContainer = document.getElementById('learnedVisualization');
        }
    } else if (exportType === 'learned') {
        // Create temporary visualization for learned graph
        targetContainer = await createTemporaryVisualization(currentResults, vizType, 'learned');
    } else if (exportType === 'groundtruth') {
        // Create temporary visualization for ground truth
        const groundTruthResults = {
            adjacency_matrix: currentResults.ground_truth_matrix,
            node_names: currentResults.node_names,
            parameters: { threshold: 0.5 }
        };
        targetContainer = await createTemporaryVisualization(groundTruthResults, vizType, 'ground_truth');
    }
    
    const plotlyDiv = targetContainer.querySelector('.plotly-graph-div');
    if (!plotlyDiv) {
        throw new Error('No visualization found to export');
    }
    
    await Plotly.downloadImage(plotlyDiv, {
        format: 'png',
        filename: filename,
        width: width,
        height: height,
        scale: 2
    });
    
    // Clean up temporary container if created
    if (exportType !== 'single' && targetContainer.id === 'tempExportContainer') {
        targetContainer.remove();
    }
    
    addLogMessage(`Exported visualization as ${filename}`, 'info');
    showAlert('Visualization exported successfully!', 'success');
}

// Export side-by-side comparison
async function exportSideBySideComparison(vizType, width, height, filename) {
    // Create a temporary container for side-by-side layout
    const tempContainer = document.createElement('div');
    tempContainer.id = 'tempSideBySideContainer';
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.width = `${width}px`;
    tempContainer.style.height = `${height}px`;
    tempContainer.style.display = 'flex';
    tempContainer.style.backgroundColor = 'white';
    
    // Create left and right containers
    const leftContainer = document.createElement('div');
    leftContainer.style.width = '50%';
    leftContainer.style.height = '100%';
    leftContainer.style.position = 'relative';
    
    const rightContainer = document.createElement('div');
    rightContainer.style.width = '50%';
    rightContainer.style.height = '100%';
    rightContainer.style.position = 'relative';
    
    tempContainer.appendChild(leftContainer);
    tempContainer.appendChild(rightContainer);
    document.body.appendChild(tempContainer);
    
    try {
        // Create learned visualization
        if (vizType === 'graph') {
            createGraphVisualization(leftContainer, currentResults, 'Learned Graph', 'learned');
        } else {
            createHeatmapVisualization(leftContainer, currentResults, 'Learned Matrix', 'learned');
        }
        
        // Create ground truth visualization
        const groundTruthResults = {
            adjacency_matrix: currentResults.ground_truth_matrix,
            node_names: currentResults.node_names,
            parameters: { threshold: 0.5 }
        };
        
        if (vizType === 'graph') {
            createGraphVisualization(rightContainer, groundTruthResults, 'Ground Truth Graph', 'ground_truth');
        } else {
            createHeatmapVisualization(rightContainer, groundTruthResults, 'Ground Truth Matrix', 'ground_truth');
        }
        
        // Wait for visualizations to render
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Convert to canvas and download
        await exportContainerAsImage(tempContainer, filename, width, height);
        
    } finally {
        // Clean up
        tempContainer.remove();
    }
    
    addLogMessage(`Exported side-by-side comparison as ${filename}`, 'info');
    showAlert('Side-by-side comparison exported successfully!', 'success');
}

// Create temporary visualization container
async function createTemporaryVisualization(results, vizType, type) {
    const tempContainer = document.createElement('div');
    tempContainer.id = 'tempExportContainer';
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.width = '800px';
    tempContainer.style.height = '600px';
    document.body.appendChild(tempContainer);
    
    if (vizType === 'graph') {
        createGraphVisualization(tempContainer, results, type === 'learned' ? 'Learned Graph' : 'Ground Truth Graph', type);
    } else {
        createHeatmapVisualization(tempContainer, results, type === 'learned' ? 'Learned Matrix' : 'Ground Truth Matrix', type);
    }
    
    // Wait for visualization to render
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return tempContainer;
}

// Export container as image using html2canvas fallback
async function exportContainerAsImage(container, filename, width, height) {
    // Try to use html2canvas if available, otherwise fall back to Plotly export
    if (typeof html2canvas !== 'undefined') {
        try {
            const canvas = await html2canvas(container, {
                width: width,
                height: height,
                scale: 2,
                backgroundColor: 'white'
            });
            
            // Create download link
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
            
        } catch (error) {
            console.warn('html2canvas failed, falling back to individual exports');
            throw error;
        }
    } else {
        // Fallback: export each Plotly chart individually
        const plotlyDivs = container.querySelectorAll('.plotly-graph-div');
        for (let i = 0; i < plotlyDivs.length; i++) {
            const div = plotlyDivs[i];
            const individualFilename = filename.replace('.png', `_part${i + 1}.png`);
            await Plotly.downloadImage(div, {
                format: 'png',
                filename: individualFilename,
                width: width / plotlyDivs.length,
                height: height,
                scale: 2
            });
        }
    }
}

// Generate export filename
function generateExportFilename(exportType, vizType) {
    const dataset = currentDataset ? currentDataset.dataset_name : 'dataset';
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    
    let typeStr = '';
    switch (exportType) {
        case 'single':
            typeStr = currentViewMode === 'single' ? 'learned' : 'comparison';
            break;
        case 'learned':
            typeStr = 'learned';
            break;
        case 'groundtruth':
            typeStr = 'groundtruth';
            break;
        case 'sidebyside':
            typeStr = 'comparison';
            break;
    }
    
    return `${dataset}_${vizType}_${typeStr}_${timestamp}.png`;
}

// Save Results Functionality
function saveResults() {
    if (!currentResults) {
        showAlert('No results to save. Run the algorithm first.', 'warning');
        return;
    }
    
    // Populate save modal with current result summary
    populateResultSummary();
    
    // Generate default name
    const defaultName = generateDefaultResultName();
    document.getElementById('resultName').value = defaultName;
    document.getElementById('resultDescription').value = '';
    
    // Show save modal
    const modal = new bootstrap.Modal(document.getElementById('saveResultsModal'));
    modal.show();
}

function populateResultSummary() {
    const summaryDiv = document.getElementById('resultSummaryPreview');
    
    const summary = `
        <div class="row">
            <div class="col-md-6">
                <strong>Dataset:</strong> ${currentDataset ? currentDataset.dataset_name : 'Unknown'}<br>
                <strong>Algorithm:</strong> ${currentResults.algorithm.name}<br>
                <strong>Runtime:</strong> ${currentResults.runtime.toFixed(2)}s
            </div>
            <div class="col-md-6">
                <strong>Learned Edges:</strong> ${currentResults.learned_edges}<br>
                <strong>Nodes:</strong> ${currentResults.node_names.length}<br>
                ${currentResults.metrics ? `<strong>F1-Score:</strong> ${currentResults.metrics.f1_score.toFixed(3)}` : '<em>No ground truth</em>'}
            </div>
        </div>
    `;
    
    summaryDiv.innerHTML = summary;
}

function generateDefaultResultName() {
    const dataset = currentDataset ? currentDataset.dataset_name : 'dataset';
    const algorithm = currentResults.algorithm.name.replace(' ', '_').toLowerCase();
    const timestamp = new Date().toLocaleString().replace(/[^\w\s]/g, '_').replace(/\s+/g, '_');
    return `${dataset}_${algorithm}_${timestamp}`;
}

async function performSaveResults() {
    const resultName = document.getElementById('resultName').value.trim();
    const description = document.getElementById('resultDescription').value.trim();
    
    if (!resultName) {
        showAlert('Please enter a name for the result.', 'warning');
        return;
    }
    
    try {
        const saveData = {
            name: resultName,
            description: description,
            result_data: currentResults,
            dataset_info: currentDataset,
            saved_date: new Date().toISOString(),
            view_mode: currentViewMode,
            visualization_type: currentVisualizationType
        };
        
        const response = await fetch('/api/save_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(saveData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('Results saved successfully!', 'success');
            addLogMessage(`Results saved as "${resultName}"`, 'info');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('saveResultsModal'));
            modal.hide();
            
        } else {
            showAlert(`Error saving results: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Save error:', error);
        showAlert('Failed to save results.', 'danger');
    }
}

// Saved Results Management
async function showSavedResults() {
    const modal = new bootstrap.Modal(document.getElementById('savedResultsModal'));
    modal.show();
    
    // Load saved results
    await loadSavedResults();
    
    // Set up search and filter
    setupResultsSearch();
}

async function loadSavedResults() {
    const gridContainer = document.getElementById('savedResultsGrid');
    const loadingDiv = document.getElementById('loadingResults');
    
    try {
        loadingDiv.style.display = 'block';
        
        const response = await fetch('/api/saved_results');
        const result = await response.json();
        
        if (result.success) {
            renderSavedResults(result.results);
        } else {
            gridContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    Error loading saved results: ${result.error}
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Load error:', error);
        gridContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-1"></i>
                Error loading saved results: ${error.message}
            </div>
        `;
    } finally {
        loadingDiv.style.display = 'none';
    }
}

// Function for the tab version
async function loadSavedResultsTab() {
    const gridContainer = document.getElementById('savedResultsTabGrid');
    const loadingDiv = document.getElementById('loadingSavedResults');
    
    if (!gridContainer) {
        console.error('savedResultsTabGrid element not found');
        return;
    }
    
    try {
        if (loadingDiv) {
            loadingDiv.style.display = 'block';
        }
        
        const response = await fetch('/api/saved_results');
        const result = await response.json();
        
        if (result.success) {
            renderSavedResultsTab(result.results);
        } else {
            gridContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    Error loading saved results: ${result.error}
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Load error:', error);
        gridContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-1"></i>
                Error loading saved results: ${error.message}
            </div>
        `;
    } finally {
        if (loadingDiv) {
            loadingDiv.style.display = 'none';
        }
    }
}

function renderSavedResults(results) {
    const gridContainer = document.getElementById('savedResultsGrid');
    
    if (results.length === 0) {
        gridContainer.innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-folder-open fa-3x mb-3 text-muted"></i>
                <h5>No Saved Results</h5>
                <p class="text-muted">Run some algorithms and save the results to see them here.</p>
            </div>
        `;
        return;
    }
    
    let html = '<div class="row">';
    
    results.forEach(result => {
        const savedDate = new Date(result.saved_date).toLocaleDateString();
        const metrics = result.result_data.metrics;
        
        html += `
            <div class="col-lg-4 col-md-6 mb-4" data-result-id="${result.id}">
                <div class="card h-100 result-card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="mb-0">${result.name}</h6>
                            <div class="form-check">
                                <input class="form-check-input result-checkbox" type="checkbox" value="${result.id}">
                            </div>
                        </div>
                        <small class="text-muted">${savedDate}</small>
                    </div>
                    <div class="card-body">
                        <div class="mb-2">
                            <strong>Dataset:</strong> ${result.dataset_info ? result.dataset_info.dataset_name : 'Unknown'}<br>
                            <strong>Algorithm:</strong> ${result.result_data.algorithm.name}<br>
                            <strong>Runtime:</strong> ${result.result_data.runtime.toFixed(2)}s<br>
                            <strong>Edges:</strong> ${result.result_data.learned_edges}
                        </div>
                        
                        ${metrics ? `
                            <div class="mb-2">
                                <small class="text-muted">Performance Metrics:</small><br>
                                <small>F1: ${metrics.f1_score.toFixed(3)} | Precision: ${metrics.precision.toFixed(3)} | Recall: ${metrics.recall.toFixed(3)}</small>
                            </div>
                        ` : ''}
                        
                        ${result.description ? `
                            <div class="mb-2">
                                <small class="text-muted">${result.description}</small>
                            </div>
                        ` : ''}
                    </div>
                    <div class="card-footer">
                        <div class="btn-group w-100" role="group">
                            <button class="btn btn-primary btn-sm" onclick="loadSavedResult('${result.id}')">
                                <i class="fas fa-eye me-1"></i>View
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="duplicateResult('${result.id}')">
                                <i class="fas fa-copy me-1"></i>Duplicate
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="deleteSingleResult('${result.id}')">
                                <i class="fas fa-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    gridContainer.innerHTML = html;
    
    // Set up checkbox event handlers
    setupCheckboxHandlers();
}

function renderSavedResultsTab(results) {
    const gridContainer = document.getElementById('savedResultsTabGrid');
    
    if (results.length === 0) {
        gridContainer.innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-folder-open fa-3x mb-3 text-muted"></i>
                <h5>No Saved Results</h5>
                <p class="text-muted">Run some algorithms and save the results to see them here.</p>
            </div>
        `;
        return;
    }
    
    let html = '<div class="row">';
    
    results.forEach(result => {
        const savedDate = new Date(result.saved_date).toLocaleDateString();
        const metrics = result.result_data.metrics;
        
        html += `
            <div class="col-lg-4 col-md-6 mb-4" data-result-id="${result.id}">
                <div class="card h-100 result-card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="mb-0">${result.name}</h6>
                            <div class="form-check">
                                <input class="form-check-input result-checkbox-tab" type="checkbox" value="${result.id}">
                            </div>
                        </div>
                        <small class="text-muted">${savedDate}</small>
                    </div>
                    <div class="card-body">
                        <div class="mb-2">
                            <strong>Dataset:</strong> ${result.dataset_info ? result.dataset_info.dataset_name : 'Unknown'}<br>
                            <strong>Algorithm:</strong> ${result.result_data.algorithm.name}<br>
                            <strong>Runtime:</strong> ${result.result_data.runtime.toFixed(2)}s<br>
                            <strong>Edges:</strong> ${result.result_data.learned_edges}
                        </div>
                        
                        ${metrics ? `
                            <div class="mb-2">
                                <small class="text-muted">Performance Metrics:</small><br>
                                <small>F1: ${metrics.f1_score.toFixed(3)} | Precision: ${metrics.precision.toFixed(3)} | Recall: ${metrics.recall.toFixed(3)}</small>
                            </div>
                        ` : ''}
                        
                        ${result.description ? `
                            <div class="mb-2">
                                <small class="text-muted">${result.description}</small>
                            </div>
                        ` : ''}
                    </div>
                    <div class="card-footer">
                        <div class="btn-group w-100" role="group">
                            <button class="btn btn-primary btn-sm" onclick="loadSavedResultToMain('${result.id}')">
                                <i class="fas fa-eye me-1"></i>View
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="duplicateResult('${result.id}')">
                                <i class="fas fa-copy me-1"></i>Duplicate
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="deleteSingleResultTab('${result.id}')">
                                <i class="fas fa-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    gridContainer.innerHTML = html;
    
    // Set up checkbox event handlers for tab
    setupCheckboxHandlersTab();
}

function setupCheckboxHandlersTab() {
    const checkboxes = document.querySelectorAll('.result-checkbox-tab');
    const deleteBtn = document.getElementById('deleteSelectedSavedBtn');
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const selectedCount = document.querySelectorAll('.result-checkbox-tab:checked').length;
            deleteBtn.disabled = selectedCount === 0;
            deleteBtn.textContent = selectedCount > 0 ? `Delete Selected (${selectedCount})` : 'Delete Selected';
        });
    });
}

function setupCheckboxHandlers() {
    const checkboxes = document.querySelectorAll('.result-checkbox');
    const deleteBtn = document.getElementById('deleteSelectedBtn');
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const selectedCount = document.querySelectorAll('.result-checkbox:checked').length;
            deleteBtn.disabled = selectedCount === 0;
            deleteBtn.textContent = selectedCount > 0 ? `Delete Selected (${selectedCount})` : 'Delete Selected';
        });
    });
}

function setupResultsSearch() {
    const searchInput = document.getElementById('resultsSearch');
    const filterSelect = document.getElementById('resultsFilter');
    
    function filterResults() {
        const searchTerm = searchInput.value.toLowerCase();
        const algorithmFilter = filterSelect.value;
        const resultCards = document.querySelectorAll('.result-card');
        
        resultCards.forEach(card => {
            const cardText = card.textContent.toLowerCase();
            const matchesSearch = searchTerm === '' || cardText.includes(searchTerm);
            const matchesFilter = algorithmFilter === '' || cardText.includes(algorithmFilter);
            
            card.closest('.col-lg-4').style.display = (matchesSearch && matchesFilter) ? 'block' : 'none';
        });
    }
    
    searchInput.addEventListener('input', filterResults);
    filterSelect.addEventListener('change', filterResults);
}

async function loadSavedResult(resultId) {
    try {
        const response = await fetch(`/api/saved_result/${resultId}`);
        const result = await response.json();
        
        if (result.success) {
            // Load the result data back into the main interface
            currentResults = result.data.result_data;
            currentDataset = result.data.dataset_info;
            currentViewMode = result.data.view_mode || 'single';
            currentVisualizationType = result.data.visualization_type || 'graph';
            
            // Close saved results modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('savedResultsModal'));
            modal.hide();
            
            // Update the main interface
            displayResults(currentResults);
            
            // Update view mode and visualization type
            document.getElementById(currentViewMode === 'single' ? 'singleView' : 'compareView').checked = true;
            document.getElementById(currentVisualizationType === 'graph' ? 'graphViz' : 'heatmapViz').checked = true;
            setViewMode(currentViewMode);
            setVisualizationType(currentVisualizationType);
            
            // Switch to results tab
            const resultsTab = new bootstrap.Tab(document.querySelector('#results-tab'));
            resultsTab.show();
            
            showAlert('Result loaded successfully!', 'success');
            addLogMessage(`Loaded saved result: ${result.data.name}`, 'info');
            
        } else {
            showAlert(`Error loading result: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Load result error:', error);
        showAlert('Failed to load result.', 'danger');
    }
}

async function loadSavedResultToMain(resultId) {
    try {
        const response = await fetch(`/api/saved_result/${resultId}`);
        const result = await response.json();
        
        if (result.success) {
            // Load the result data back into the main interface
            currentResults = result.data.result_data;
            currentDataset = result.data.dataset_info;
            currentViewMode = result.data.view_mode || 'single';
            currentVisualizationType = result.data.visualization_type || 'graph';
            
            // Update the main interface
            displayResults(currentResults);
            
            // Update view mode and visualization type
            document.getElementById(currentViewMode === 'single' ? 'singleView' : 'compareView').checked = true;
            document.getElementById(currentVisualizationType === 'graph' ? 'graphViz' : 'heatmapViz').checked = true;
            setViewMode(currentViewMode);
            setVisualizationType(currentVisualizationType);
            
            // Switch to results tab
            const resultsTab = new bootstrap.Tab(document.querySelector('#results-tab'));
            resultsTab.show();
            
            showAlert('Result loaded successfully!', 'success');
            addLogMessage(`Loaded saved result: ${result.data.name}`, 'info');
            
        } else {
            showAlert(`Error loading result: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Load result error:', error);
        showAlert('Failed to load result.', 'danger');
    }
}

async function deleteSingleResultTab(resultId) {
    if (!confirm('Are you sure you want to delete this result?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/saved_result/${resultId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('Result deleted successfully!', 'success');
            await loadSavedResultsTab(); // Refresh the tab list
        } else {
            showAlert(`Error deleting result: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('Failed to delete result.', 'danger');
    }
}

async function deleteSingleResult(resultId) {
    if (!confirm('Are you sure you want to delete this result?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/saved_result/${resultId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('Result deleted successfully!', 'success');
            await loadSavedResults(); // Refresh the list
        } else {
            showAlert(`Error deleting result: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('Failed to delete result.', 'danger');
    }
}

async function deleteSelectedSavedResults() {
    const selectedIds = Array.from(document.querySelectorAll('.result-checkbox-tab:checked')).map(cb => cb.value);
    
    if (selectedIds.length === 0) {
        return;
    }
    
    if (!confirm(`Are you sure you want to delete ${selectedIds.length} result(s)?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/delete_multiple_results', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ result_ids: selectedIds })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert(`${selectedIds.length} result(s) deleted successfully!`, 'success');
            await loadSavedResultsTab(); // Refresh the tab list
        } else {
            showAlert(`Error deleting results: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('Failed to delete results.', 'danger');
    }
}

async function deleteSelectedResults() {
    const selectedIds = Array.from(document.querySelectorAll('.result-checkbox:checked')).map(cb => cb.value);
    
    if (selectedIds.length === 0) {
        return;
    }
    
    if (!confirm(`Are you sure you want to delete ${selectedIds.length} result(s)?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/delete_multiple_results', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ result_ids: selectedIds })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert(`${selectedIds.length} result(s) deleted successfully!`, 'success');
            await loadSavedResults(); // Refresh the list
        } else {
            showAlert(`Error deleting results: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('Failed to delete results.', 'danger');
    }
}

async function duplicateResult(resultId) {
    try {
        const response = await fetch(`/api/duplicate_result/${resultId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('Result duplicated successfully!', 'success');
            await loadSavedResults(); // Refresh the list
        } else {
            showAlert(`Error duplicating result: ${result.error}`, 'danger');
        }
        
    } catch (error) {
        console.error('Duplicate error:', error);
        showAlert('Failed to duplicate result.', 'danger');
    }
}


// Utility functions
function updateStatus(message, status) {
    const statusElement = document.getElementById('status');
    const indicator = document.createElement('span');
    indicator.className = `status-indicator status-${status}`;
    statusElement.innerHTML = indicator.outerHTML + message;
}

function addLogMessage(message, level = 'info') {
    const logContainer = document.getElementById('logContainer');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level}`;
    logEntry.innerHTML = `[${timestamp}] ${message}`;
    
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function clearLog() {
    document.getElementById('logContainer').innerHTML = '<div class="text-muted">Log cleared...</div>';
}

function showAlert(message, type) {
    // Create Bootstrap alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const mainContent = document.querySelector('.col-md-9');
    mainContent.insertBefore(alertDiv, mainContent.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// =================== DATASET MANAGEMENT FUNCTIONS ===================

// Show dataset upload modal
function showDatasetUpload() {
    resetModalToCreateMode();
    const modal = new bootstrap.Modal(document.getElementById('datasetUploadModal'));
    modal.show();
}

// Reset modal to create mode
function resetModalToCreateMode() {
    // Reset edit mode flags
    document.getElementById('editMode').value = 'false';
    document.getElementById('originalDatasetName').value = '';
    
    // Reset modal title and button
    document.getElementById('datasetModalTitle').innerHTML = '<i class="fas fa-database me-2"></i>Create New Dataset';
    document.getElementById('saveButtonText').textContent = 'Save Dataset';
    document.getElementById('deleteDatasetBtn').style.display = 'none';
    
    // Reset form fields
    document.getElementById('csvUpload').value = '';
    document.getElementById('bifUpload').value = '';
    document.getElementById('datasetName').value = '';
    document.getElementById('datasetName').disabled = false;
    document.getElementById('datasetDescription').value = '';
    
    // Reset CSV upload state
    document.getElementById('csvUpload').disabled = false;
    document.getElementById('csvUploadHelp').style.display = 'block';
    document.getElementById('csvEditNotice').style.display = 'none';
    
    // Hide sections
    document.getElementById('validationCard').style.display = 'none';
    document.getElementById('previewCard').style.display = 'none';
    document.getElementById('bifPreview').style.display = 'none';
    document.getElementById('datasetStats').style.display = 'none';
    document.getElementById('saveDatasetBtn').disabled = true;
    
    // Reset data
    uploadedCSVData = null;
    uploadedBIFContent = null;
}

// Validate uploaded CSV file
async function validateUploadedCSV() {
    const fileInput = document.getElementById('csvUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        document.getElementById('validationCard').style.display = 'none';
        document.getElementById('previewCard').style.display = 'none';
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/validate_dataset', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            uploadedCSVData = result.preview;
            displayValidationResults(result.validation);
            displayDatasetPreview(result.preview, result.columns);
            updateDatasetStats(result.shape, result.columns);
            
            // Auto-generate dataset name from filename
            if (!document.getElementById('datasetName').value) {
                const name = file.name.replace('.csv', '').replace(/[^a-zA-Z0-9]/g, '_');
                document.getElementById('datasetName').value = name;
            }
            
            checkSaveReadiness();
        } else {
            showAlert(`Validation failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error validating file: ${error.message}`, 'danger');
    }
}

// Display validation results
function displayValidationResults(validation) {
    const resultsDiv = document.getElementById('validationResults');
    const validationCard = document.getElementById('validationCard');
    
    let html = '';
    
    if (validation.valid) {
        html += '<div class="alert alert-success"><i class="fas fa-check-circle me-2"></i>Dataset format is valid for NOTEARS!</div>';
    } else {
        html += '<div class="alert alert-danger"><i class="fas fa-exclamation-triangle me-2"></i>Dataset has validation errors</div>';
    }
    
    if (validation.errors && validation.errors.length > 0) {
        html += '<div class="mt-2"><strong>Errors:</strong><ul class="mb-0 text-danger">';
        validation.errors.forEach(error => {
            html += `<li>${error}</li>`;
        });
        html += '</ul></div>';
    }
    
    if (validation.warnings && validation.warnings.length > 0) {
        html += '<div class="mt-2"><strong>Warnings:</strong><ul class="mb-0 text-warning">';
        validation.warnings.forEach(warning => {
            html += `<li>${warning}</li>`;
        });
        html += '</ul></div>';
    }
    
    if (validation.info) {
        html += '<div class="mt-2"><strong>Dataset Info:</strong><ul class="mb-0">';
        html += `<li>Shape: ${validation.info.shape[0]} samples × ${validation.info.shape[1]} variables</li>`;
        html += `<li>Numeric columns: ${validation.info.numeric_columns}</li>`;
        html += `<li>Missing values: ${validation.info.missing_values}</li>`;
        html += `<li>Memory usage: ${validation.info.memory_usage}</li>`;
        html += '</ul></div>';
    }
    
    resultsDiv.innerHTML = html;
    validationCard.style.display = 'block';
}

// Preview BIF file content
function previewBIF() {
    const fileInput = document.getElementById('bifUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        document.getElementById('bifPreview').style.display = 'none';
        uploadedBIFContent = null;
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedBIFContent = e.target.result;
        document.getElementById('bifContent').value = uploadedBIFContent;
        document.getElementById('bifPreview').style.display = 'block';
        checkSaveReadiness();
    };
    reader.readAsText(file);
}

// Display dataset preview in upload modal
function displayDatasetPreview(data, columns) {
    const container = document.getElementById('datasetPreviewContainer');
    const previewCard = document.getElementById('previewCard');
    
    if (!data || data.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-3">No data to preview</div>';
        return;
    }
    
    let tableHtml = `
        <table class="table table-sm table-striped dataset-table">
            <thead class="table-dark">
                <tr>
                    ${columns.map(col => `<th>${col}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
    `;
    
    data.forEach(row => {
        tableHtml += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
            tableHtml += `<td>${displayValue}</td>`;
        });
        tableHtml += '</tr>';
    });
    
    tableHtml += '</tbody></table>';
    
    if (data.length >= 10) {
        tableHtml += '<div class="text-muted text-center mt-2"><small>Showing first 10 rows</small></div>';
    }
    
    container.innerHTML = tableHtml;
    previewCard.style.display = 'block';
}

// Update dataset statistics
function updateDatasetStats(shape, columns) {
    const statsDiv = document.getElementById('datasetStats');
    const statsContent = document.getElementById('statsContent');
    
    let html = `
        <div class="row">
            <div class="col-6"><strong>Samples:</strong></div>
            <div class="col-6">${shape[0]}</div>
        </div>
        <div class="row">
            <div class="col-6"><strong>Variables:</strong></div>
            <div class="col-6">${shape[1]}</div>
        </div>
        <div class="row">
            <div class="col-6"><strong>Columns:</strong></div>
            <div class="col-6">${columns.join(', ')}</div>
        </div>
    `;
    
    statsContent.innerHTML = html;
    statsDiv.style.display = 'block';
}

// Check if all requirements are met for saving
function checkSaveReadiness() {
    const nameInput = document.getElementById('datasetName');
    const saveBtn = document.getElementById('saveDatasetBtn');
    
    const hasName = nameInput.value.trim().length > 0;
    const hasValidData = uploadedCSVData !== null;
    
    saveBtn.disabled = !(hasName && hasValidData);
}

// Show dataset upload modal in edit mode
function showDatasetUploadInEditMode(dataset) {
    // Set edit mode
    document.getElementById('editMode').value = 'true';
    document.getElementById('originalDatasetName').value = dataset.dataset_name;
    
    // Update modal title
    document.getElementById('datasetModalTitle').innerHTML = '<i class="fas fa-edit me-2"></i>Edit Dataset: ' + dataset.dataset_name;
    
    // Update save button
    document.getElementById('saveButtonText').textContent = 'Update Dataset';
    
    // Show delete button in edit mode
    document.getElementById('deleteDatasetBtn').style.display = 'block';
    
    // Fill in existing data
    document.getElementById('datasetName').value = dataset.dataset_name;
    document.getElementById('datasetName').disabled = true; // Can't change name in edit mode
    
    // Fill description if available
    if (dataset.info && dataset.info.description) {
        document.getElementById('datasetDescription').value = dataset.info.description;
    }
    
    // Show CSV upload notice and disable CSV upload
    document.getElementById('csvUpload').disabled = true;
    document.getElementById('csvUploadHelp').style.display = 'none';
    document.getElementById('csvEditNotice').style.display = 'block';
    
    // Pre-populate with existing CSV data for preview
    uploadedCSVData = dataset.preview;
    displayDatasetPreview(dataset.preview, dataset.columns);
    updateDatasetStats(dataset.shape, dataset.columns);
    document.getElementById('previewCard').style.display = 'block';
    
    // Load existing BIF file if present
    if (dataset.has_ground_truth) {
        loadExistingBIF(dataset.dataset_name);
    }
    
    // Show validation as successful since data already exists
    const validation = {
        valid: true,
        warnings: [],
        errors: [],
        info: {
            shape: dataset.shape,
            numeric_columns: dataset.columns.length,
            missing_values: 0,
            memory_usage: "N/A"
        }
    };
    displayValidationResults(validation);
    
    // Enable save button
    document.getElementById('saveDatasetBtn').disabled = false;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('datasetUploadModal'));
    modal.show();
}

// Load existing BIF file for editing
async function loadExistingBIF(datasetName) {
    try {
        const response = await fetch(`/datasets/${datasetName}/${datasetName}.bif`);
        if (response.ok) {
            const bifContent = await response.text();
            uploadedBIFContent = bifContent;
            document.getElementById('bifContent').value = bifContent;
            document.getElementById('bifPreview').style.display = 'block';
        }
    } catch (error) {
        console.log('No existing BIF file found or error loading it');
    }
}

// Save or update dataset (handles both create and edit modes)
async function saveOrUpdateDataset() {
    const datasetName = document.getElementById('datasetName').value.trim();
    const description = document.getElementById('datasetDescription').value.trim();
    const isEditMode = document.getElementById('editMode').value === 'true';
    
    if (!datasetName || !uploadedCSVData) {
        showAlert('Please provide a dataset name and valid CSV data.', 'warning');
        return;
    }
    
    try {
        let response, result;
        
        if (isEditMode) {
            // Update metadata only (description and BIF)
            const payload = {
                description: description,
                bif_content: uploadedBIFContent || ''
            };
            
            response = await fetch(`/api/update_dataset_metadata/${datasetName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
        } else {
            // Create new dataset
            const payload = {
                dataset_name: datasetName,
                description: description,
                csv_data: uploadedCSVData,
                bif_content: uploadedBIFContent || ''
            };
            
            response = await fetch('/api/save_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
        }
        
        result = await response.json();
        
        if (result.success) {
            showAlert(result.message, 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('datasetUploadModal'));
            modal.hide();
            
            // Reset modal to create mode
            resetModalToCreateMode();
            
            // Refresh dataset list
            await loadAvailableDatasets();
            
            // Select the dataset
            document.getElementById('datasetSelect').value = datasetName;
            if (currentDataset && currentDataset.dataset_name === datasetName) {
                await loadSelectedDataset(); // Refresh if it's the current dataset
            }
            
        } else {
            showAlert(`Error ${isEditMode ? 'updating' : 'saving'} dataset: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error ${isEditMode ? 'updating' : 'saving'} dataset: ${error.message}`, 'danger');
    }
}

// Edit selected dataset - opens create modal in edit mode
async function editSelectedDataset() {
    const select = document.getElementById('datasetSelect');
    const datasetName = select.value;
    
    if (!datasetName) {
        showAlert('Please select a dataset to edit.', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/load_dataset/${datasetName}`);
        const result = await response.json();
        
        if (result.success) {
            showDatasetUploadInEditMode(result);
        } else {
            showAlert(`Error loading dataset for editing: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error loading dataset: ${error.message}`, 'danger');
    }
}

// Delete (archive) selected dataset
async function deleteSelectedDataset() {
    const datasetName = document.getElementById('datasetName').value.trim();
    
    if (!datasetName) {
        showAlert('No dataset to delete.', 'warning');
        return;
    }
    
    // Confirmation dialog
    if (!confirm(`Are you sure you want to delete dataset "${datasetName}"?\n\nThe dataset will be moved to the trashcan and can be restored later.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete_dataset/${datasetName}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert(result.message, 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('datasetUploadModal'));
            modal.hide();
            
            // Reset modal to create mode
            resetModalToCreateMode();
            
            // Refresh dataset list
            await loadAvailableDatasets();
            
            // Clear selection if this was the current dataset
            if (currentDataset && currentDataset.dataset_name === datasetName) {
                document.getElementById('datasetSelect').value = '';
                currentDataset = null;
                document.getElementById('datasetInfo').style.display = 'none';
                document.getElementById('dataPreview').innerHTML = '<div class="text-center text-muted py-5"><i class="fas fa-database fa-3x mb-3"></i><p>Select a dataset to view preview</p></div>';
            }
            
        } else {
            showAlert(`Error deleting dataset: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error deleting dataset: ${error.message}`, 'danger');
    }
}

// Show trashcan modal
async function showTrashcan() {
    const modal = new bootstrap.Modal(document.getElementById('trashcanModal'));
    modal.show();
    
    // Load archived datasets
    await loadArchivedDatasets();
}

// Load archived datasets
async function loadArchivedDatasets() {
    const container = document.getElementById('archivedDatasetsList');
    
    try {
        const response = await fetch('/api/archived_datasets');
        const result = await response.json();
        
        if (result.success) {
            if (result.datasets.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted py-4">
                        <i class="fas fa-trash fa-3x mb-3"></i>
                        <p>No deleted datasets found.</p>
                        <small>Deleted datasets will appear here and can be restored.</small>
                    </div>
                `;
            } else {
                let html = '<div class="list-group">';
                
                result.datasets.forEach(dataset => {
                    const deletedDate = new Date(dataset.deleted_date).toLocaleString();
                    html += `
                        <div class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="flex-grow-1">
                                    <h6 class="mb-1">${dataset.name}</h6>
                                    <p class="mb-1 text-muted">${dataset.description || 'No description'}</p>
                                    <small class="text-muted">
                                        <i class="fas fa-clock me-1"></i>Deleted: ${deletedDate}
                                        <span class="ms-3"><i class="fas fa-nodes me-1"></i>${dataset.nodes} nodes</span>
                                        <span class="ms-2"><i class="fas fa-database me-1"></i>${dataset.samples} samples</span>
                                    </small>
                                </div>
                                <div class="ms-3">
                                    <button class="btn btn-success btn-sm" onclick="restoreDataset('${dataset.name}')">
                                        <i class="fas fa-undo me-1"></i>Restore
                                    </button>
                                    <button class="btn btn-danger btn-sm ms-1" onclick="permanentlyDeleteDataset('${dataset.name}')">
                                        <i class="fas fa-times me-1"></i>Delete Forever
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                container.innerHTML = html;
            }
        } else {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    Error loading archived datasets: ${result.error}
                </div>
            `;
        }
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-1"></i>
                Error loading archived datasets: ${error.message}
            </div>
        `;
    }
}

// Restore dataset from archive
async function restoreDataset(datasetName) {
    if (!confirm(`Restore dataset "${datasetName}" from the trashcan?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/restore_dataset/${datasetName}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert(result.message, 'success');
            
            // Refresh archived datasets list
            await loadArchivedDatasets();
            
            // Refresh main dataset list
            await loadAvailableDatasets();
            
        } else {
            showAlert(`Error restoring dataset: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error restoring dataset: ${error.message}`, 'danger');
    }
}

// Permanently delete dataset
async function permanentlyDeleteDataset(datasetName) {
    if (!confirm(`PERMANENTLY DELETE dataset "${datasetName}"?\n\nThis action cannot be undone! The dataset will be completely removed.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/permanently_delete_dataset/${datasetName}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert(result.message, 'warning');
            
            // Refresh archived datasets list
            await loadArchivedDatasets();
            
        } else {
            showAlert(`Error permanently deleting dataset: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error permanently deleting dataset: ${error.message}`, 'danger');
    }
}


// Add event listener for dataset name input
document.addEventListener('DOMContentLoaded', function() {
    const nameInput = document.getElementById('datasetName');
    if (nameInput) {
        nameInput.addEventListener('input', checkSaveReadiness);
    }
});