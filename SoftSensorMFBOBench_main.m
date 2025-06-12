%@author: Paolo Conti, Johannes Lips

% Outer loop for combining ANN Multi-Fidelity Regression with Bayesian Optimization (BO) for Hierarchy Determination in MATLAB
% The pyExec location should be the same as the shebang in the inner loop python script

%% === set up the python environment ===
rng(42); %set random seed
warning('off', 'all'); % suppress warnings

pyExec = 'C:\Users\~\Anaconda3\envs\py310\python.exe'; %location of python version to use
pyRoot = fileparts(pyExec);
p = getenv('PATH');
p = strsplit(p, ';');
%to find the following components: start the active environment in conda, then execute %PATH%  
addToPath = {
    pyRoot
    fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
    fullfile(pyRoot, 'Library', 'usr', 'bin')
    fullfile(pyRoot, 'Library', 'bin')
    fullfile(pyRoot, 'Scripts')
    fullfile(pyRoot, 'bin')
    };

pe = pyenv(Version=pyExec);

%% === Bayesian Optimization hyperparameters ===
acqfnName = "expected-improvement";
determObjective = false;
numLFs = 6; %number of low fidelity datasets, should match the number of elements in the python variable original_fidelity_order (number of measurement signals available)
numiter = 25; %number of iterations of the BO 

% the hyperparameters eta to be optimized:
% number of LF signals to use
vars = [optimizableVariable('numUsedInputs', [1, numLFs], 'Type', 'integer')];
% for each measurement signal the fidelity score
for i = 1:numLFs
        vars = [vars, ...
            optimizableVariable(['fidelityX', num2str(i-1)], [0, numLFs], 'Type', 'integer'), ...
        ];
end

% loss function and constraitn function from function definitions below
loss_function = @(x) loss_function_helper(x); 
xconstraint_function = @(x) xconstraint_function_helper(x);

%% === run BO ===

% initial run
results = bayesopt(loss_function, vars, ...
    "AcquisitionFunctionName",acqfnName, ...
    "XConstraintSatisfiabilitySamples",2e6, ...
    "GPActiveSetSize",2e3, ...
    "NumCoupledConstraints",0, ...
    'XConstraintFcn', xconstraint_function, ...
    "Verbose",1, ...
    "NumSeedPoints", 1, ...
    'IsObjectiveDeterministic', determObjective, ...
    'MaxObjectiveEvaluations', 1, ...
    'PlotFcn',{@plotObjective,@plotMinObjective, @plotElapsedTime}, ...
    'OutputFcn', {@saveToFile}, ...
    'SaveFileName', './tempres' ...
    );

% iteratively evaluate more points, this allows to expand the traces with
% implicitly observed points after each BO iteration (this is not necessary
% for a simpler setup, then just set MaxObjectiveEvaluations = numiter and
% evaluate bayesopt() only once)
for i = 1:numiter
    close('all')
    contX = results.XTrace;
    contY = results.ObjectiveTrace;
    % expand the traces of the observed points with the implicitly evaluated points
    if i ~= 1
        [contX, contY] = expand_traces(contX, contY, results.UserDataTrace{end,1});
    end
    
    results = bayesopt(loss_function, vars, ...
        "AcquisitionFunctionName",acqfnName, ...
        "XConstraintSatisfiabilitySamples",2e6, ...
        "GPActiveSetSize",2e3, ...
        "NumCoupledConstraints",0, ...
        'XConstraintFcn', xconstraint_function, ...
        "Verbose",1, ...
        "InitialX", contX, ...
        "InitialObjective", contY, ...
        "NumSeedPoints", 1, ...
        'IsObjectiveDeterministic', determObjective, ...
        'MaxObjectiveEvaluations', numel(contY) + 1, ... 
        'PlotFcn',{@plotObjective,@plotMinObjective, @plotElapsedTime}, ...
        'OutputFcn', {@saveToFile}, ...
        'SaveFileName', './tempres' ...
        );
end

%% === get the optimum hyperparameters ===
OptX = results.XAtMinEstimatedObjective;
OptNumLF = OptX{1,1}; %numLF always returns 1 higher than what is returned (off-by-one)
% get used sensors as it is done in loss_function_helper
OptFidelityScore = OptX{1,2:end};
[sorted, sortedIndices] = sort(OptFidelityScore);
OptSensorOrder = sortedIndices; %(no off-by-one for python since this is for depicting and user interpretation: first signal is LF1!)
OptUnusedSensors = OptSensorOrder(sorted >= OptNumLF);
OptSensorOrder = OptSensorOrder(sorted < OptNumLF);
OptSensorOrder = flip(OptSensorOrder);

disp('=== Optimum found by BO ===')
disp(['Number of inputs: ' num2str(OptNumLF)])
disp(['Selected sensors in hierarchy: [' num2str(OptSensorOrder) ']'])
disp(['Discarded sensors: [' num2str(sort(OptUnusedSensors)) ']'])


%% === store the results ===
[contX, contY] = expand_traces(contX, contY, results.UserDataTrace{end,1});
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['simResults_', timestamp, '.mat'];
save(filename, 'OptX', 'contX', 'contY')

%% functions
function [objective,coupledconstraints, userdata] = loss_function_helper(x)
    % loss function (objective function) of the BO
    % example:
    % [numUsedInputs, x0fid, x1fid, x2fid, x3fid] = 
    % [2 0 1 3 4] should give [1 0] as fidelity_order_labels for the inner python loop
    % [2 3 1 0 4] it should give [1 2] as fidelity_order_labels
    % also mind the 0-indexing of python vs 1-indexing of MATLAB

    % fidelityScore of LF signals
    fidelityScore = table2array(removevars(x, 'numUsedInputs'));
    [sorted, sortedIndices] = sort(fidelityScore);
    sensorOrder = sortedIndices-1;
    % numUsedInputs: only keep inputs which were ranked beloy the numUsedInputs
    sensorOrder = sensorOrder(sorted < x.('numUsedInputs'));
    % now invert this: the one with the lowest value has the highest: fidelity and should be last!
    sensorOrder = flip(sensorOrder);

    % run the inner loop
    [all_goodness] = pyrunfile("SoftSensorMFBOBench_inner.py", ["all_goodness_val"], fidelity_order_labels=sensorOrder, externalInput=1);

    objective = double(all_goodness);
    objective = objective(end); %note that this will be deleted and overwritten with the userdata all_goodness when using the expand_traces function
    coupledconstraints = []; %there are no coupled constraints in this problem
    userdata = {sensorOrder, all_goodness};
end

function isFeasible = xconstraint_function_helper(XTable)
    % xconstraint: check if selected point is feasible hyperparameter,
    % i.e., check if it is a permutation of numLFs

    % sort the rows and check if they are the same as 0:numel, then take
    % the sum and compare with numel
    [sorted, sortedIndices] = sort(table2array(XTable),2);
    isFeasible = ismember(sorted,0:(size(sorted,2)-1),'rows');
end

function [contX, contY] = expand_traces(contX, contY, lastUserData)
    % starting from the last point that was evaluated by BO (taken from
    % the userdata), add all intermediately evaluated points, as well as
    % all permutations that lead to the same selection of LF signals.
    % This significantly speeds up the BO procedure!
    
    % remove the last row from contX and contY
    contX = contX(1:(end-1),:);
    contY = contY(1:(end-1));

    % extract the used inputs and all_goodness from the userData
    fullSensorOrder = lastUserData{1, 1};
    all_goodness = double(lastUserData{1, 2});
    numTotalInputs = size(contX,2) - 1;

    % iterate over each evaluated combination and find all permutations that would lead to the same result
    for i = 1:numel(fullSensorOrder)
        goodness = all_goodness(i);
        numUsedInputs = i; %numUsedInputs = number of sensors or datasets being used
        sensorOrder = fullSensorOrder(1:i);
        sensorOrder = flip(sensorOrder); %flip it back, so that highest fidelity score is first instead of last
        sensorRank = 0:(i-1); %corresponding ranking
        %number of permutations = number of rows we need = number of
        %elements that are not given as inputs
        numUnusedSensors =  numTotalInputs - numUsedInputs;
        numRows = factorial(numUnusedSensors);
        unusedSensorRank = (i:(numTotalInputs - 1)) + 1;
        allPerms = perms(unusedSensorRank);
        
        % create a table with the implicitly evaluated points
        variableNames = contX.Properties.VariableNames;
        variableTypes = repmat({'double'}, [1, numel(variableNames)]);
        newX = table('Size', [numRows, numel(variableNames)], 'VariableNames', variableNames, 'VariableTypes', variableTypes);
        newX{:,1} = numUsedInputs*ones([numRows 1]);
        k = 1; %first column of allPerms that has not been used yet
        for j = 1:numTotalInputs
            if ismember(j,sensorOrder+1)
                newX{:,j+1} = sensorRank(sensorOrder+1==j)*ones([numRows 1]);
            else
                newX{:,j+1} = allPerms(:,k);
                k = k+1;
            end
        end
    
    % add implicitly evaluated points to the result traces
    contX = [contX; newX];
    contY = [contY; goodness*ones([numRows 1])];
    end
end