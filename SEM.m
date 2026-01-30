%% ========================================================================
% REQUIRED FILES:
% 1. dominance_diff.mat
%    - Contains spatial probability difference data for dominance classification.
% 2. Domain_grid.mat
%    - Contains the spatial grid definitions for climate zones.
% 3. SM_VPD_rh_ssr_t2m_tp_at_GPP_40.txt
%    - The raw time-series data file.
%
% TXT FILE CONTENT FORMAT:
% The text file contains the following columns with headers:
% - Longitude: Geographic longitude (e.g., 80.75)
% - Latitude: Geographic latitude (e.g., 36.25)
% - Time: Time identifier (e.g., May-00)
% - GPP: Gross Primary Productivity (Vegetation productivity metric)
% - SM: Soil Moisture
% - VPD: Vapor Pressure Deficit
% - rh: Relative Humidity
% - ssr: Surface Solar Radiation
% - t2m: Temperature at 2 meters
% - tp: Total Precipitation
% - domain: Climate zone label (e.g., A, B, C...)
%
% CODE FUNCTIONALITY:
% 1. Loads spatial data (dominance_diff, Domain_grid) required for calculating dominance types.
% 2. Loads the raw time-series data containing GPP, SM, VPD, etc.
% 3. Matches each data point to the dominance grid based on Longitude and Latitude 
%    to determine if it is "SM-Dominated" or "VPD-Dominated".
% 4. Groups the data by "Climate Zone" (A-G) and "Dominance Type".
% 5. Performs VIF (Variance Inflation Factor) tests for each group to check for multicollinearity.
% 6. If VIF is acceptable, constructs a Structural Equation Model (SEM) and 
%    calculates fit indices (Chi2, CFI, TLI, RMSEA, SRMR).
% 7. Summarizes and displays the Fit Indices table and VIF table in the Command Window.
% 8. Automatically draws and displays SEM path coefficient diagrams for both 
%    SM-dominated and VPD-dominated regions based on the calculation results.
%
%% ========================================================================

%% ========================================================================
% Script: Calculate overall SEM model fit indices, VIF, and plot path diagrams
%
% Dependencies:
% - dominance_diff.mat
% - Domain_grid.mat
% - SM_VPD_rh_ssr_t2m_tp_at_GPP_40.txt
%
%% ========================================================================

%% 1. Environment Cleanup and Path Settings
clear;
close all;
clc;
fprintf('>> Starting SEM fit indices calculation, VIF test, and plotting script...\n');

% --- [Please modify your file paths here] ---
data_folder = 'H:\';
dominance_diff_file = fullfile(data_folder, '5_Pdiff_40_all_thresholds', 'dominance_diff.mat');
domain_grid_file = fullfile(data_folder, '7_Pearson', 'Domain_grid.mat');
raw_data_file = 'SM_VPD_rh_ssr_t2m_tp_at_GPP_40.txt'; % Assuming this file is in the current working directory or MATLAB path

%% 2. Load Required Data
fprintf('>> Loading data...\n');
try
    % Load probability difference and partition grid data
    load(dominance_diff_file, 'dominance_diff');
    load(domain_grid_file, 'Domain_grid');

    % Load raw time-series data
    opts = detectImportOptions(raw_data_file, 'FileType', 'text');
    opts.VariableNames = {'Longitude', 'Latitude', 'Time', 'GPP', 'SM', 'VPD', 'rh', 'ssr', 't2m', 'tp', 'domain'};
    rawData = readtable(raw_data_file, opts);

    % Compatibility handling: If 'domain' column is numeric, convert to character labels
    if isnumeric(rawData.domain)
        rawData.domain = cellstr(char(rawData.domain + 64)); % 1->'A', 2->'B', etc.
    end

    fprintf('Data loaded successfully. Total %d raw records.\n', height(rawData));
catch ME
    error('Failed to load data file! Please check if file path and name are correct.\n\nError details: %s', ME.message);
end

%% 3. Data Preprocessing and Dominance Type Matching
fprintf('>> Matching data points with dominance grid...\n');

% --- Unify Grid Dimensions ---
[rows_domain, cols_domain] = size(Domain_grid);
[rows_diff, cols_diff] = size(dominance_diff);
common_rows = min(rows_domain, rows_diff);
common_cols = min(cols_domain, cols_diff);
dominance_diff_subset = dominance_diff(1:common_rows, 1:common_cols);

% --- Define Grid Geographic Coordinates Parameters ---
[grid_rows, grid_cols] = size(dominance_diff_subset);
lon_start = 73.75;      % Grid westernmost longitude
lon_step = 0.5;         % Grid longitude step
lat_north_edge = 53.25; % Grid northernmost latitude
lat_step_original = -0.5; % Original latitude step
lat_south_edge = lat_north_edge + grid_rows * lat_step_original; % Calculate southernmost latitude

% --- Start Matching ---
rawData.dominance_type = NaN(height(rawData), 1); % 1 for SM, -1 for VPD

% Calculate column indices (Longitude)
col_indices = floor((rawData.Longitude - lon_start) / lon_step) + 1;

% Calculate row indices (Latitude, considering "top-south bottom-north" logic)
lat_step_corrected = 0.5;
row_indices = floor((rawData.Latitude - lat_south_edge) / lat_step_corrected) + 1;

% Filter valid indices within grid range
valid_mask = col_indices >= 1 & col_indices <= grid_cols & ...
             row_indices >= 1 & row_indices <= grid_rows;

% Extract valid indices and assign dominance types
if any(valid_mask)
    valid_rows = row_indices(valid_mask);
    valid_cols = col_indices(valid_mask);
    
    linear_indices = sub2ind(size(dominance_diff_subset), valid_rows, valid_cols);
    point_dominance_values = dominance_diff_subset(linear_indices);
    
    dominance_classification = NaN(size(point_dominance_values));
    dominance_classification(point_dominance_values > 0) = 1;  % SM Dominated
    dominance_classification(point_dominance_values <= 0) = -1; % VPD Dominated
    
    rawData.dominance_type(valid_mask) = dominance_classification;
else
    warning('Warning: No data points found within the valid grid range.');
end

fprintf('Dominance type classification complete: SM-Dominated %d, VPD-Dominated %d, Unassigned %d.\n', ...
    sum(rawData.dominance_type == 1, 'omitnan'), ...
    sum(rawData.dominance_type == -1, 'omitnan'), ...
    sum(isnan(rawData.dominance_type)));

%% 4. Calculate SEM Fit Indices and VIF by Zone and Dominance Type
fprintf('>> Starting SEM fit and VIF calculation by zone...\n');

zone_labels = {'A','B','C','D','E','F','G'};
num_zones = numel(zone_labels);

% Initialize cell arrays to store SEM results
sem_results_sm_dom = cell(num_zones + 1, 1); % +1 for China
sem_results_vpd_dom = cell(num_zones + 1, 1);

% Initialize cell arrays to store VIF results
vif_rh_sm_dom = cell(num_zones + 1, 1);
vif_vpd_sm_dom = cell(num_zones + 1, 1);
vif_gpp_sm_dom = cell(num_zones + 1, 1);

vif_rh_vpd_dom = cell(num_zones + 1, 1);
vif_vpd_vpd_dom = cell(num_zones + 1, 1);
vif_gpp_vpd_dom = cell(num_zones + 1, 1);

% --- Loop through A-G seven climate zones ---
for k = 1:num_zones
    current_label = zone_labels{k};
    fprintf(' -- Processing Climate Zone: %s\n', current_label);
    
    zone_data = rawData(strcmp(rawData.domain, current_label), :);
    
    % Calculate for SM-Dominated areas
    data_sm_dom = zone_data(zone_data.dominance_type == 1, :);
    [sem_results_sm_dom{k}, vif_rh_sm_dom{k}, vif_vpd_sm_dom{k}, vif_gpp_sm_dom{k}] = ...
        calculate_sem_metrics(data_sm_dom, current_label, 'SM-Dominated');
    
    % Calculate for VPD-Dominated areas
    data_vpd_dom = zone_data(zone_data.dominance_type == -1, :);
    [sem_results_vpd_dom{k}, vif_rh_vpd_dom{k}, vif_vpd_vpd_dom{k}, vif_gpp_vpd_dom{k}] = ...
        calculate_sem_metrics(data_vpd_dom, current_label, 'VPD-Dominated');
end

% --- Process the entire China region ---
fprintf(' -- Processing entire region: China\n');
all_data_sm_dom = rawData(rawData.dominance_type == 1, :);
[sem_results_sm_dom{end}, vif_rh_sm_dom{end}, vif_vpd_sm_dom{end}, vif_gpp_sm_dom{end}] = ...
    calculate_sem_metrics(all_data_sm_dom, 'China', 'SM-Dominated');

all_data_vpd_dom = rawData(rawData.dominance_type == -1, :);
[sem_results_vpd_dom{end}, vif_rh_vpd_dom{end}, vif_vpd_vpd_dom{end}, vif_gpp_vpd_dom{end}] = ...
    calculate_sem_metrics(all_data_vpd_dom, 'China', 'VPD-Dominated');

fprintf('SEM fit indices and VIF calculations completed for all zones.\n');

%% 5. Organize and Display SEM Fit Results
fprintf('\n================== Model Fit Indices Summary ==================\n');
row_names = [zone_labels, {'China'}];
col_names = {'Chi2_SM', 'CFI_SM', 'TLI_SM', 'RMSEA_SM', 'SRMR_SM', 'SampleSize_SM', ...
             'Chi2_VPD', 'CFI_VPD', 'TLI_VPD', 'RMSEA_VPD', 'SRMR_VPD', 'SampleSize_VPD'};
fit_data = NaN(numel(row_names), numel(col_names));

% Fill the result matrix
for i = 1:numel(row_names)
    % SM Dominated results
    if ~isempty(sem_results_sm_dom{i}) && isfield(sem_results_sm_dom{i}, 'CFI') && ~isnan(sem_results_sm_dom{i}.CFI)
        fit_data(i, 1) = sem_results_sm_dom{i}.Chi2;
        fit_data(i, 2) = sem_results_sm_dom{i}.CFI;
        fit_data(i, 3) = sem_results_sm_dom{i}.TLI;
        fit_data(i, 4) = sem_results_sm_dom{i}.RMSEA;
        fit_data(i, 5) = sem_results_sm_dom{i}.SRMR;
        fit_data(i, 6) = sem_results_sm_dom{i}.N;
    end
    
    % VPD Dominated results
    if ~isempty(sem_results_vpd_dom{i}) && isfield(sem_results_vpd_dom{i}, 'CFI') && ~isnan(sem_results_vpd_dom{i}.CFI)
        fit_data(i, 7) = sem_results_vpd_dom{i}.Chi2;
        fit_data(i, 8) = sem_results_vpd_dom{i}.CFI;
        fit_data(i, 9) = sem_results_vpd_dom{i}.TLI;
        fit_data(i, 10) = sem_results_vpd_dom{i}.RMSEA;
        fit_data(i, 11) = sem_results_vpd_dom{i}.SRMR;
        fit_data(i, 12) = sem_results_vpd_dom{i}.N;
    end
end

% Create and display table
fit_table = array2table(fit_data, 'RowNames', row_names, 'VariableNames', col_names);
disp(fit_table);

fprintf('\nFit Indices Interpretation:\n');
fprintf(' Chi2 (Chi-Square): Smaller is better, dependent on degrees of freedom\n');
fprintf(' CFI (Comparative Fit Index): > 0.90 Acceptable, > 0.95 Good\n');
fprintf(' TLI (Tucker-Lewis Index): > 0.90 Acceptable, > 0.95 Good\n');
fprintf(' RMSEA (Root Mean Square Error): < 0.08 Acceptable, < 0.05 Good\n');
fprintf(' SRMR (Standardized Root Mean Sq): < 0.08 Acceptable, < 0.05 Good\n');
fprintf(' SampleSize: Effective sample size used for model calculation\n');
fprintf('========================================================\n\n');

%% 6. Organize and Display VIF Summary Results
fprintf('\n================== VIF (Variance Inflation Factor) Test Summary ==================\n');
row_names_vif = [zone_labels, {'China'}]';

% --- Table 1: VIF for Model rh ~ tp + SM ---
vif_rh_cols = {'tp_SM_Dom', 'SM_SM_Dom', 'tp_VPD_Dom', 'SM_VPD_Dom'};
vif_rh_data = NaN(numel(row_names_vif), numel(vif_rh_cols));

for i = 1:numel(row_names_vif)
    % SM Dominated
    if ~isempty(vif_rh_sm_dom{i})
        vif_rh_data(i, 1) = vif_rh_sm_dom{i}.VIF(strcmp(vif_rh_sm_dom{i}.Variable, 'z_tp'));
        vif_rh_data(i, 2) = vif_rh_sm_dom{i}.VIF(strcmp(vif_rh_sm_dom{i}.Variable, 'z_SM'));
    end
    % VPD Dominated
    if ~isempty(vif_rh_vpd_dom{i})
        vif_rh_data(i, 3) = vif_rh_vpd_dom{i}.VIF(strcmp(vif_rh_vpd_dom{i}.Variable, 'z_tp'));
        vif_rh_data(i, 4) = vif_rh_vpd_dom{i}.VIF(strcmp(vif_rh_vpd_dom{i}.Variable, 'z_SM'));
    end
end
vif_rh_table = array2table(vif_rh_data, 'RowNames', row_names_vif, 'VariableNames', vif_rh_cols);
fprintf('\n--- VIF Values for Model rh ~ tp + SM ---\n');
disp(vif_rh_table);

% --- Table 2: VIF for Model VPD ~ t2m + rh ---
vif_vpd_cols = {'t2m_SM_Dom', 'rh_SM_Dom', 't2m_VPD_Dom', 'rh_VPD_Dom'};
vif_vpd_data = NaN(numel(row_names_vif), numel(vif_vpd_cols));

for i = 1:numel(row_names_vif)
    % SM Dominated
    if ~isempty(vif_vpd_sm_dom{i})
        vif_vpd_data(i, 1) = vif_vpd_sm_dom{i}.VIF(strcmp(vif_vpd_sm_dom{i}.Variable, 'z_t2m'));
        vif_vpd_data(i, 2) = vif_vpd_sm_dom{i}.VIF(strcmp(vif_vpd_sm_dom{i}.Variable, 'z_rh'));
    end
    % VPD Dominated
    if ~isempty(vif_vpd_vpd_dom{i})
        vif_vpd_data(i, 3) = vif_vpd_vpd_dom{i}.VIF(strcmp(vif_vpd_vpd_dom{i}.Variable, 'z_t2m'));
        vif_vpd_data(i, 4) = vif_vpd_vpd_dom{i}.VIF(strcmp(vif_vpd_vpd_dom{i}.Variable, 'z_rh'));
    end
end
vif_vpd_table = array2table(vif_vpd_data, 'RowNames', row_names_vif, 'VariableNames', vif_vpd_cols);
fprintf('\n--- VIF Values for Model VPD ~ t2m + rh ---\n');
disp(vif_vpd_table);

% --- Table 3: VIF for Model GPP ~ SM + ssr + VPD ---
vif_gpp_cols = {'SM_SM_Dom', 'ssr_SM_Dom', 'VPD_SM_Dom', 'SM_VPD_Dom', 'ssr_VPD_Dom', 'VPD_VPD_Dom'};
vif_gpp_data = NaN(numel(row_names_vif), numel(vif_gpp_cols));

for i = 1:numel(row_names_vif)
    % SM Dominated
    if ~isempty(vif_gpp_sm_dom{i})
        vif_gpp_data(i, 1) = vif_gpp_sm_dom{i}.VIF(strcmp(vif_gpp_sm_dom{i}.Variable, 'z_SM'));
        vif_gpp_data(i, 2) = vif_gpp_sm_dom{i}.VIF(strcmp(vif_gpp_sm_dom{i}.Variable, 'z_ssr'));
        vif_gpp_data(i, 3) = vif_gpp_sm_dom{i}.VIF(strcmp(vif_gpp_sm_dom{i}.Variable, 'z_VPD'));
    end
    % VPD Dominated
    if ~isempty(vif_gpp_vpd_dom{i})
        vif_gpp_data(i, 4) = vif_gpp_vpd_dom{i}.VIF(strcmp(vif_gpp_vpd_dom{i}.Variable, 'z_SM'));
        vif_gpp_data(i, 5) = vif_gpp_vpd_dom{i}.VIF(strcmp(vif_gpp_vpd_dom{i}.Variable, 'z_ssr'));
        vif_gpp_data(i, 6) = vif_gpp_vpd_dom{i}.VIF(strcmp(vif_gpp_vpd_dom{i}.Variable, 'z_VPD'));
    end
end
vif_gpp_table = array2table(vif_gpp_data, 'RowNames', row_names_vif, 'VariableNames', vif_gpp_cols);
fprintf('\n--- VIF Values for Model GPP ~ SM + ssr + VPD ---\n');
disp(vif_gpp_table);

fprintf('Note: VIF > 5.0 usually indicates multicollinearity requiring attention. NaN indicates insufficient data or calculation skipped due to high VIF.\n');
fprintf('========================================================================\n\n');

%% 7. Organize Plotting Data (Link Calculation Results to Plotting)
fprintf('>> Preparing plotting data...\n');

% Variable Initialization
num_paths = 10;
num_regions_plot = 8; % A, B, C, D, E, F, G, China
coeffs_sm = NaN(num_paths, num_regions_plot);
pvals_sm = NaN(num_paths, num_regions_plot);
coeffs_vpd = NaN(num_paths, num_regions_plot);
pvals_vpd = NaN(num_paths, num_regions_plot);

% Extract data into matrix (for easier plotting loop)
for k = 1:num_regions_plot
    % SM Dominated
    if ~isempty(sem_results_sm_dom{k}) && isfield(sem_results_sm_dom{k}, 'coeffs') && ~all(isnan(sem_results_sm_dom{k}.coeffs))
        coeffs_sm(:, k) = sem_results_sm_dom{k}.coeffs';
        pvals_sm(:, k) = sem_results_sm_dom{k}.pvals';
    end
    % VPD Dominated
    if ~isempty(sem_results_vpd_dom{k}) && isfield(sem_results_vpd_dom{k}, 'coeffs') && ~all(isnan(sem_results_vpd_dom{k}.coeffs))
        coeffs_vpd(:, k) = sem_results_vpd_dom{k}.coeffs';
        pvals_vpd(:, k) = sem_results_vpd_dom{k}.pvals';
    end
end
fprintf('Data preparation complete.\n');

%% 8. Plot SEM Path Diagrams
fprintf('>> Starting to plot SEM path diagrams...\n');

% --- Define Plot Labels and Font ---
plot_zone_labels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'China'};
chinese_font = 'Times New Roman'; % Adjust based on OS: Windows use 'Microsoft YaHei' or 'Times New Roman'

% --- [Fig 4]: Plot SM-Dominated SEM Path Diagrams ---
fprintf('Generating [Fig 4 (Path Diagram)]: SM-Dominated Zones...\n');
fig4 = figure('Position', [50, 50, 1625, 500], 'Name', 'SEM Path Diagram for SM-Dominated Zones (8 Plots)', 'Color', 'w');
for k = 1:num_regions_plot
    ax = subplot(2, 4, k);
    current_coeffs = coeffs_sm(:, k);
    current_pvals = pvals_sm(:, k);
    current_title = plot_zone_labels{k};
    
    % Call core plotting function
    draw_sem_path_diagram(ax, current_coeffs, current_pvals, current_title, chinese_font);
end

% Add overall title
annotation(fig4, 'textbox', [0.1, 0.92, 0.8, 0.05], ...
    'String', 'SM-Dominated: SEM Path Models for Climate Zones and Overall', ...
    'FontName', chinese_font, ...
    'FontSize', 10.5, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'LineStyle', 'none');

fprintf('[Fig 4] Generation complete.\n');

% --- [Fig 5]: Plot VPD-Dominated SEM Path Diagrams ---
fprintf('Generating [Fig 5 (Path Diagram)]: VPD-Dominated Zones...\n');
fig5 = figure('Position', [50, 50, 1625, 500], 'Name', 'SEM Path Diagram for VPD-Dominated Zones (8 Plots)', 'Color', 'w');
for k = 1:num_regions_plot
    ax = subplot(2, 4, k);
    current_coeffs = coeffs_vpd(:, k);
    current_pvals = pvals_vpd(:, k);
    current_title = plot_zone_labels{k};
    
    % Call core plotting function
    draw_sem_path_diagram(ax, current_coeffs, current_pvals, current_title, chinese_font);
end

% Add overall title
annotation(fig5, 'textbox', [0.1, 0.92, 0.8, 0.05], ...
    'String', 'VPD-Dominated: SEM Path Models for Climate Zones and Overall', ...
    'FontName', chinese_font, ...
    'FontSize',10.5, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'LineStyle', 'none');

fprintf('[Fig 5] Generation complete.\n');
fprintf('\n>>>>> All tasks (Calculation, Test, Plotting) completed.\n');

%% ========================================================================
% % [Core Helper Function]
% % Function: Calculate SEM path coefficients and fit indices based on data subset
% %========================================================================
function [results, vif_rh, vif_vpd, vif_gpp] = calculate_sem_metrics(data_subset, zone_name, dominance_name)

    % Initialize output structure
    results = struct('coeffs', NaN(1,10), 'pvals', NaN(1,10), ...
        'Chi2', NaN, 'df', NaN, 'CFI', NaN, 'TLI', NaN, ...
        'RMSEA', NaN, 'SRMR', NaN, 'N', 0);
    
    % Initialize VIF output tables
    vif_rh = []; vif_vpd = []; vif_gpp = [];

    % --- 1. Data Preprocessing ---
    MIN_SAMPLES = 30; % Set a reasonable minimum sample size
    if height(data_subset) < MIN_SAMPLES
        return; % Return if data is insufficient
    end
    
    var_names = {'GPP', 'SM', 'VPD', 'rh', 'ssr', 't2m', 'tp'};
    tbl = data_subset(:, var_names);
    tbl = rmmissing(tbl); % Remove rows containing any NaN values
    N = height(tbl);      % Effective sample size
    
    if N < MIN_SAMPLES
        return;
    end
    results.N = N;
    
    z_tbl = varfun(@zscore, tbl); % Data standardization (Z-score)
    z_tbl.Properties.VariableNames = strcat('z_', var_names);
    p = numel(var_names); % Number of variables

    % --- 2. VIF Test: Check for Multicollinearity ---
    VIF_THRESHOLD = 5.0; % Set VIF threshold, usually 5 or 10

    % Test Model rh ~ tp + SM
    predictors_rh = z_tbl(:, {'z_tp', 'z_SM'});
    vif_rh = calculate_vif(predictors_rh);
    if any(vif_rh.VIF > VIF_THRESHOLD)
        fprintf('Warning: Severe multicollinearity (VIF > %.1f) detected in rh model for [%s, %s]. Analysis skipped.\n', zone_name, dominance_name, VIF_THRESHOLD);
        results.CFI = NaN; return;
    end

    % Test Model VPD ~ t2m + rh
    predictors_vpd = z_tbl(:, {'z_t2m', 'z_rh'});
    vif_vpd = calculate_vif(predictors_vpd);
    if any(vif_vpd.VIF > VIF_THRESHOLD)
        fprintf('Warning: Severe multicollinearity (VIF > %.1f) detected in VPD model for [%s, %s]. Analysis skipped.\n', zone_name, dominance_name, VIF_THRESHOLD);
        results.CFI = NaN; return;
    end

    % Test Model GPP ~ SM + ssr + VPD
    predictors_gpp = z_tbl(:, {'z_SM', 'z_ssr', 'z_VPD'});
    vif_gpp = calculate_vif(predictors_gpp);
    if any(vif_gpp.VIF > VIF_THRESHOLD)
        fprintf('Warning: Severe multicollinearity (VIF > %.1f) detected in GPP model for [%s, %s]. Analysis skipped.\n', zone_name, dominance_name, VIF_THRESHOLD);
        results.CFI = NaN; return;
    end

    % --- 3. Estimate Path Coefficients and R-squared using fitlm (if VIF passed) ---
    coeffs = NaN(1, 10);
    pvals = NaN(1, 10);
    r_squared = NaN(p, 1); % Store R-squared for each endogenous variable

    try
        % Path 1: tp -> SM
        mdl_sm = fitlm(z_tbl.z_tp, z_tbl.z_SM);
        coeffs(1) = mdl_sm.Coefficients.Estimate(2); 
        pvals(1) = mdl_sm.Coefficients.pValue(2);
        r_squared(2) = mdl_sm.Rsquared.Ordinary; % SM is var 2

        % Path 2: tp -> ssr
        mdl_ssr = fitlm(z_tbl.z_tp, z_tbl.z_ssr);
        coeffs(2) = mdl_ssr.Coefficients.Estimate(2); 
        pvals(2) = mdl_ssr.Coefficients.pValue(2);
        r_squared(5) = mdl_ssr.Rsquared.Ordinary; % ssr is var 5

        % Paths 3, 5: tp -> rh, SM -> rh
        mdl_rh = fitlm([z_tbl.z_tp, z_tbl.z_SM], z_tbl.z_rh);
        coeffs(3) = mdl_rh.Coefficients.Estimate(2); pvals(3) = mdl_rh.Coefficients.pValue(2);
        coeffs(5) = mdl_rh.Coefficients.Estimate(3); pvals(5) = mdl_rh.Coefficients.pValue(3);
        r_squared(4) = mdl_rh.Rsquared.Ordinary; % rh is var 4

        % Path 7: ssr -> t2m
        mdl_t2m = fitlm(z_tbl.z_ssr, z_tbl.z_t2m);
        coeffs(7) = mdl_t2m.Coefficients.Estimate(2); 
        pvals(7) = mdl_t2m.Coefficients.pValue(2);
        r_squared(6) = mdl_t2m.Rsquared.Ordinary; % t2m is var 6

        % Paths 8, 9: t2m -> VPD, rh -> VPD
        mdl_vpd = fitlm([z_tbl.z_t2m, z_tbl.z_rh], z_tbl.z_VPD);
        coeffs(8) = mdl_vpd.Coefficients.Estimate(2); pvals(8) = mdl_vpd.Coefficients.pValue(2);
        coeffs(9) = mdl_vpd.Coefficients.Estimate(3); pvals(9) = mdl_vpd.Coefficients.pValue(3);
        r_squared(3) = mdl_vpd.Rsquared.Ordinary; % VPD is var 3

        % Paths 4, 6, 10: SM -> GPP, ssr -> GPP, VPD -> GPP
        mdl_gpp = fitlm([z_tbl.z_SM, z_tbl.z_ssr, z_tbl.z_VPD], z_tbl.z_GPP);
        coeffs(4) = mdl_gpp.Coefficients.Estimate(2); pvals(4) = mdl_gpp.Coefficients.pValue(2);
        coeffs(6) = mdl_gpp.Coefficients.Estimate(3); pvals(6) = mdl_gpp.Coefficients.pValue(3);
        coeffs(10) = mdl_gpp.Coefficients.Estimate(4); pvals(10) = mdl_gpp.Coefficients.pValue(4);
        r_squared(1) = mdl_gpp.Rsquared.Ordinary; % GPP is var 1

        results.coeffs = coeffs;
        results.pvals = pvals;

    catch ME
        fprintf('Error: An error occurred during model fitting (fitlm) for [%s, %s]: %s\n', zone_name, dominance_name, ME.message);
        results.CFI = NaN; return;
    end

    % --- 4. Calculate Model Implied Covariance Matrix (Sigma_implied) ---
    % Variable Order: 1:GPP, 2:SM, 3:VPD, 4:rh, 5:ssr, 6:t2m, 7:tp
    B = zeros(p, p); % Endogenous -> Endogenous
    B(1, 2) = coeffs(4);  % SM -> GPP
    B(1, 5) = coeffs(6);  % ssr -> GPP
    B(1, 3) = coeffs(10); % VPD -> GPP
    B(3, 6) = coeffs(8);  % t2m -> VPD
    B(3, 4) = coeffs(9);  % rh -> VPD
    B(4, 2) = coeffs(5);  % SM -> rh
    B(6, 5) = coeffs(7);  % ssr -> t2m

    Gamma = zeros(p, 1); % Exogenous -> Endogenous
    Gamma(2, 1) = coeffs(1); % tp -> SM
    Gamma(5, 1) = coeffs(2); % tp -> ssr
    Gamma(4, 1) = coeffs(3); % tp -> rh

    Phi = 1; % Exogenous variable covariance (tp variance is 1)
    
    psi_diag = 1 - r_squared; % Residual variance = 1 - R^2
    psi_diag(7) = 0; % Residual variance for exogenous variable 'tp' is 0
    Psi = diag(psi_diag);
    
    I = eye(p);
    inv_I_minus_B = inv(I - B);
    
    Sigma_implied = inv_I_minus_B * (Gamma * Phi * Gamma' + Psi) * inv_I_minus_B';
    Sigma_implied(7,7) = 1; % Ensure exogenous variable variance is 1

    % --- 5. Calculate Observed Covariance Matrix (S) ---
    S = cov(table2array(z_tbl));

    % --- 6. Calculate Chi-Square Statistic ---
    F_ml = log(det(Sigma_implied)) - log(det(S)) + trace(S / Sigma_implied) - p;
    chi2_model = (N - 1) * F_ml;
    
    % Degrees of Freedom df = (Observable Covariances) - (Free Parameters)
    df_model = (p*(p+1)/2) - 16;
    results.Chi2 = chi2_model;
    results.df = df_model;

    % --- 7. Calculate CFI and TLI ---
    S_diag = diag(diag(S));
    F_null = log(det(S_diag)) - log(det(S)) + trace(S / S_diag) - p;
    chi2_null = (N - 1) * F_null;
    df_null = (p*(p+1)/2) - p;

    % Calculate CFI
    d_model = max(0, chi2_model - df_model);
    d_null = max(0, chi2_null - df_null);
    if d_null > 1e-9
        results.CFI = 1 - (d_model / d_null);
    else
        results.CFI = 1;
    end

    % Calculate TLI
    if df_null > 0 && df_model > 0
        tli_numerator = (chi2_null / df_null) - (chi2_model / df_model);
        tli_denominator = (chi2_null / df_null) - 1;
        if tli_denominator > 1e-9
            results.TLI = tli_numerator / tli_denominator;
        else
            results.TLI = 1;
        end
    else
        results.TLI = NaN;
    end

    % --- 8. Calculate RMSEA ---
    if df_model > 0
        results.RMSEA = sqrt(max(0, (chi2_model/df_model - 1) / (N - 1)));
    end

    % --- 9. Calculate SRMR ---
    std_residuals = (S - Sigma_implied);
    sum_squares = 0; count = 0;
    for i = 1:p
        for j = 1:i
            denominator = sqrt(S(i,i) * S(j,j));
            if denominator > 1e-9
                standardized_residual = std_residuals(i,j) / denominator;
                sum_squares = sum_squares + standardized_residual^2;
                count = count + 1;
            end
        end
    end
    if count > 0
        results.SRMR = sqrt(sum_squares / count);
    else
        results.SRMR = NaN;
    end
end

%% ========================================================================
% % [VIF Calculation Helper Function]
% % Function: Calculate Variance Inflation Factor (VIF) for a given set of predictors
% %========================================================================
function vif_table = calculate_vif(predictor_table)
    % Input: A table containing multiple predictor variables
    % Output: A table containing VIF values for each variable
    
    var_names = predictor_table.Properties.VariableNames;
    num_vars = numel(var_names);
    vif_values = zeros(num_vars, 1);
    
    for i = 1:num_vars
        % Use the i-th variable as the response, others as predictors
        response_var = predictor_table.(var_names{i});
        predictor_vars_temp = predictor_table;
        predictor_vars_temp.(var_names{i}) = []; % Remove current response from predictors
        
        % Fit linear model
        mdl = fitlm(predictor_vars_temp, response_var);
        
        % Extract R-squared and calculate VIF
        rsquared = mdl.Rsquared.Ordinary;
        if rsquared > 0.999999
            vif_values(i) = Inf;
        else
            vif_values(i) = 1 / (1 - rsquared);
        end
    end
    vif_table = table(var_names', vif_values, 'VariableNames', {'Variable', 'VIF'});
end

%% ========================================================================
% % [Plotting Helper Function]
% % Function: Draw SEM path diagram on specified axis (ax)
% %========================================================================
function draw_sem_path_diagram(ax, coeffs, pvals, zone_title, font_name)
    axes(ax); % Activate specified subplot
    cla;      % Clear current subplot content
    hold on;

    % Check if input data is valid, if all NaN display message
    if all(isnan(coeffs))
        text(0.5, 0.5, 'No Valid Data', 'HorizontalAlignment', 'center', ...
             'FontName', font_name, 'FontSize',10.5, 'Color', [0.5 0.5 0.5]);
        text(0.5, -0.05, zone_title, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'top', 'FontName', font_name, 'FontSize', 10.5, 'FontWeight', 'bold');
        axis off;
        return;
    end

    % --- Node Positions and Appearance Definitions ---
    pos = struct(...
        'TP', [0.5, 0.7],  'SM', [0.15, 0.9], 'RH', [0.85, 0.9], ...
        'SSR', [0.3, 0.4], 'TM', [0.7, 0.4],  'VPD', [0.85, 0.1], ...
        'GPP', [0.15, 0.1]);
    
    box_w = 0.2; box_h = 0.1;

    % --- Variable Name List ---
    vars = {'TP', 'SM', 'RH', 'SSR', 'TM', 'VPD', 'GPP'};
    var_names_map = containers.Map(vars, vars); % Map variable names to display names

    % Draw all node boxes
    for i = 1:length(vars)
        v = vars{i};
        p = pos.(v);
        rectangle('Position', [p(1)-box_w/2, p(2)-box_h/2, box_w, box_h], ...
            'EdgeColor', 'k', 'LineWidth', 1.2, 'FaceColor', [0.95 0.95 0.95], 'Curvature', [0.1 0.1]);
        text(p(1), p(2), var_names_map(v), 'HorizontalAlignment','center', ...
             'VerticalAlignment', 'middle', 'FontSize',10.5, 'FontWeight','bold', 'FontName',font_name);
    end

    % --- Path Drawing ---
    % coeffs index mapping:
    % 1:TP->SM, 2:TP->SSR, 3:TP->RH, 4:SM->GPP, 5:SM->RH
    % 6:SSR->GPP, 7:SSR->TM, 8:TM->VPD, 9:RH->VPD, 10:VPD->GPP
    draw_arrow_with_label('TP', 'SM', coeffs(1), pvals(1), [-0.06,-0.06]);
    draw_arrow_with_label('TP', 'SSR', coeffs(2), pvals(2), [-0.08, 0]);
    draw_arrow_with_label('TP', 'RH', coeffs(3), pvals(3), [0.06, -0.06]);
    draw_arrow_with_label('SM', 'GPP', coeffs(4), pvals(4), [-0.08, 0]);
    draw_arrow_with_label('SM', 'RH', coeffs(5), pvals(5), [0, 0.05]);
    draw_arrow_with_label('SSR', 'GPP', coeffs(6), pvals(6), [0.1, 0]);
    draw_arrow_with_label('SSR', 'TM', coeffs(7), pvals(7), [0, 0.05]);
    draw_arrow_with_label('TM', 'VPD', coeffs(8), pvals(8), [-0.1, 0]);
    draw_arrow_with_label('RH', 'VPD', coeffs(9), pvals(9), [0.1, 0]);
    draw_arrow_with_label('VPD', 'GPP', coeffs(10), pvals(10),[0, -0.06]);

    hold off;
    axis off;
    xlim([0 1]);
    ylim([0 1]);

    % Add title below subplot
    text(0.5, -0.05, zone_title, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontName', font_name, 'FontSize', 10.5, 'FontWeight', 'bold');

    % --- Nested Helper Function: Draw Arrow with Label ---
    function draw_arrow_with_label(start_var, end_var, coeff_val, p_val, label_pos_offset)
        if isnan(coeff_val)
            return;
        end
        
        color_pos = [0, 0, 1]; % Blue
        color_neg = [1, 0, 0]; % Red
        arrow_color = color_pos;
        if coeff_val < 0, arrow_color = color_neg; end
        
        p1 = pos.(start_var);
        p2 = pos.(end_var);
        
        vec = p2 - p1;
        dist = norm(vec);
        if dist == 0, dist = 1; end
        unit_vec = vec / dist;
        
        % Arrow gap settings
        arrow_gap = 0.02;
        if strcmp(start_var, 'TP') && (strcmp(end_var, 'SM') || strcmp(end_var, 'SSR') || strcmp(end_var, 'RH'))
            arrow_gap = 0.035;
        end
        
        % Calculate intersection with rectangle edge
        theta = atan2(unit_vec(2) * box_w, unit_vec(1) * box_h);
        x_offset = box_w/2 * cos(theta);
        y_offset = box_h/2 * sin(theta);
        
        start_point_on_edge = p1 + [sign(vec(1))*abs(x_offset), sign(vec(2))*abs(y_offset)];
        end_point_on_edge = p2 - [sign(vec(1))*abs(x_offset), sign(vec(2))*abs(y_offset)];
        
        start_point = start_point_on_edge + unit_vec * arrow_gap;
        end_point = end_point_on_edge - unit_vec * arrow_gap;
        
        % Draw Arrow
        h_fig = gcf;
        ax_pos = get(ax, 'Position');
        x_fig_start = ax_pos(1) + start_point(1) * ax_pos(3);
        y_fig_start = ax_pos(2) + start_point(2) * ax_pos(4);
        x_fig_end = ax_pos(1) + end_point(1) * ax_pos(3);
        y_fig_end = ax_pos(2) + end_point(2) * ax_pos(4);
        
        annotation(h_fig, 'arrow', [x_fig_start, x_fig_end], [y_fig_start, y_fig_end], ...
            'Color', arrow_color, ...
            'LineWidth', max(0.5, 3.5 * abs(coeff_val)), ...
            'HeadStyle', 'vback2', 'HeadWidth', 8, 'HeadLength', 8);
        
        % Draw Label
        label_x = mean([start_point(1), end_point(1)]) + label_pos_offset(1);
        label_y = mean([start_point(2), end_point(2)]) + label_pos_offset(2);
        
        stars_str = '';
        if p_val < 0.01
            stars_str = '**';
        elseif p_val < 0.05
            stars_str = '*';
        end
        
        label_text = sprintf('%.2f%s', coeff_val, stars_str);
        
        text(label_x, label_y, label_text, 'HorizontalAlignment', 'center', ...
             'FontSize', 10.5, 'FontWeight', 'bold', 'Color', arrow_color, ...
             'FontName', font_name, 'BackgroundColor', 'none', 'Margin', 1);
    end
end
