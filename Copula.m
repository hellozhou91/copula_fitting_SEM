% =========================================================================
% SCRIPT DESCRIPTION
% =========================================================================
% REQUIRED FILES:
% 1. GPP Data File (.txt): Contains Longitude, Latitude, Time, GPP data.
% 2. SM Data File (.txt): Contains Longitude, Latitude, Time, SM data.
% 3. VPD Data File (.txt): Contains Longitude, Latitude, Time, VPD data.
%
% FUNCTIONALITY:
% 1. Data Loading & Preprocessing: Merges GPP, SM, and VPD data based on location and time.
% 2. Marginal Distribution Selection: Selects the best distribution for each variable using AIC.
% 3. Copula Fitting: Fits 2D Copulas (SM-GPP, VPD-GPP) and 3D C-Vine Copulas.
% 4. Probability Calculation: Calculates conditional triggering probabilities (PSM, PVPD, PSM_VPD).
% 5. Validation: Performs PIT (Probability Integral Transform) tests.
% 6. Visualization: Generates maps with specific color gradients and formatting.
% 7. Data Export: Saves results to MAT files and CSV tables.
% =========================================================================

clear; clc; close all;

tic; % Start timer

%% ========================================================================
% 1. User Configuration
% =========================================================================
disp('--- 1. User Configuration ---');

% --- File Paths (Please modify according to your actual situation) ---
GPP_FILE_PATH = 'H:\NIRv_GPP.txt';
SM_FILE_PATH = 'H:\ERA5_SM.txt';
VPD_FILE_PATH = 'H:\ERA5_VPD.txt';

OUTPUT_DIR = pwd; % Results will be saved in the current folder

% --- Representative Pixels ---
% Group 1: Good fit pixels for SM-GPP PIT
sm_gpp_pixels = [
    101.5, 26.5
    100.5, 29.5
    87.0, 29.5
    84.0, 46.5
    103.0, 26.5
    100.5, 27.0
];

% Group 2: Good fit pixels for VPD-GPP PIT
vpd_gpp_pixels = [
    87.0, 43.5
    95.5, 31.0
    95.0, 32.5
    95.5, 33.5
    96.5, 35.5
    95.5, 35.5
];

% Combine all representative pixels
representative_pixels = [sm_gpp_pixels; vpd_gpp_pixels];

% --- Model Selection Configuration ---
% 7 Candidate distributions
marginal_dists = {'Normal', 'Lognormal', 'Gamma', 'Weibull', ...
                  'GeneralizedExtremeValue', 'BirnbaumSaunders', 'tLocationScale'};

copula_families = {'Gaussian', 't', 'Clayton', 'Gumbel', 'Frank'};

% --- Calculation Parameters ---
MIN_DATA_POINTS = 30;
GRID_RESOLUTION = 0.5;

GPP_PERCENTILE = 40;
SM_PERCENTILE = 10;
VPD_PERCENTILE = 90;

% Selection Criterion (AIC only)
SELECTION_CRITERION = 'AIC';

% --- PIT Good Fit Threshold ---
PIT_GOOD_FIT_THRESHOLD = 0.05;

% --- Parallel Computing Settings ---
USE_PARALLEL = true;
MAX_PARALLEL_WORKERS = 4;
SHOW_PROGRESS = true;
PROGRESS_INTERVAL = 50;

%% ========================================================================
% 2. Clean up Parallel Pool History
% =========================================================================
disp('--- 2. Cleaning up Parallel Pool History ---');
try
    myCluster = parcluster('Processes');
    if ~isempty(myCluster.Jobs)
        fprintf('Found %d old parallel pool jobs, cleaning up...\n', length(myCluster.Jobs));
        for job_idx = 1:length(myCluster.Jobs)
            try, delete(myCluster.Jobs(job_idx)); catch, end
        end
        fprintf('Old jobs cleaned.\n');
    end
catch ME
    fprintf('Warning during parallel pool cleanup: %s\n', ME.message);
end

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
    disp('Existing parallel pool closed');
end

%% ========================================================================
% 3. Load and Preprocess Data
% =========================================================================
disp('--- 3. Loading and Preprocessing Data ---');
try
    fprintf('Loading GPP data...\n');
    GPP_data = readtable(GPP_FILE_PATH);
    
    fprintf('Loading SM data...\n');
    SM_data = readtable(SM_FILE_PATH);
    if ismember('sw', SM_data.Properties.VariableNames)
        SM_data = renamevars(SM_data, 'sw', 'SM');
    end
    
    fprintf('Loading VPD data...\n');
    VPD_data = readtable(VPD_FILE_PATH);
    
    fprintf('Merging data...\n');
    join_keys = {'Longitude', 'Latitude', 'Time'};
    merged_data_temp = innerjoin(GPP_data, SM_data, 'Keys', join_keys);
    all_data = innerjoin(merged_data_temp, VPD_data, 'Keys', join_keys);
    
    clear GPP_data SM_data VPD_data merged_data_temp;
    
    valid_idx = all(table2array(all_data(:, {'GPP','SM','VPD'})) > 0 & ...
                    ~isnan(table2array(all_data(:, {'GPP','SM','VPD'}))), 2);
    all_data = all_data(valid_idx, :);
    
    fprintf('Data loading and merging complete. Valid rows: %d\n', height(all_data));
    data_info = whos('all_data');
    fprintf('Data memory usage: %.2f MB\n', data_info.bytes / 1024^2);
    
catch ME
    error('Data loading failed: %s', ME.message);
end

%% ========================================================================
% 4. Initialize Grid and Storage Variables
% =========================================================================
disp('--- 4. Initializing Grid ---');
lon_grid = min(all_data.Longitude):GRID_RESOLUTION:max(all_data.Longitude);
lat_grid = min(all_data.Latitude ):GRID_RESOLUTION:max(all_data.Latitude );

lon_grid = round(lon_grid * 2) / 2;
lat_grid = round(lat_grid * 2) / 2;

[n_lat, n_lon] = deal(length(lat_grid), length(lon_grid));
fprintf('Grid size: %d (Lat) x %d (Lon) = %d grid cells\n', n_lat, n_lon, n_lat*n_lon);

% Result Matrices
PSM_grid = NaN(n_lat, n_lon);
PVPD_grid = NaN(n_lat, n_lon);
PSM_VPD_grid = NaN(n_lat, n_lon);

best_dist_sm_grid = cell(n_lat, n_lon);
best_dist_vpd_grid = cell(n_lat, n_lon);
best_dist_gpp_grid = cell(n_lat, n_lon);

dist_selection_results_sm_grid = cell(n_lat, n_lon);
dist_selection_results_vpd_grid = cell(n_lat, n_lon);
dist_selection_results_gpp_grid = cell(n_lat, n_lon);

best_copula_sm_gpp_grid = cell(n_lat, n_lon);
best_copula_vpd_gpp_grid = cell(n_lat, n_lon);

best_copula_aic_sm_gpp_grid = NaN(n_lat, n_lon);
best_copula_aic_vpd_gpp_grid = NaN(n_lat, n_lon);

lower_tail_sm_gpp_grid = NaN(n_lat, n_lon);
upper_tail_sm_gpp_grid = NaN(n_lat, n_lon);
lower_tail_vpd_gpp_grid = NaN(n_lat, n_lon);
upper_tail_vpd_gpp_grid = NaN(n_lat, n_lon);

% Vine Related (Only keeping C-Vine)
best_vine_structure_grid = cell(n_lat, n_lon); % Stores specific C-Vine structure info
c_vine_aic_grid = NaN(n_lat, n_lon);
c_vine_bic_grid = NaN(n_lat, n_lon);

pit_data_grid = cell(n_lat, n_lon);

% PIT Statistics
N_SM_GPP_grid = NaN(n_lat, n_lon);
Mean_SM_GPP_grid= NaN(n_lat, n_lon);
Std_SM_GPP_grid = NaN(n_lat, n_lon);
KS_p_SM_GPP_grid= NaN(n_lat, n_lon);

N_VPD_GPP_grid = NaN(n_lat, n_lon);
Mean_VPD_GPP_grid= NaN(n_lat, n_lon);
Std_VPD_GPP_grid = NaN(n_lat, n_lon);
KS_p_VPD_GPP_grid= NaN(n_lat, n_lon);

%% ========================================================================
% 5. Prepare Parallel Calculation Data
% =========================================================================
disp('--- 5. Preparing Parallel Calculation Data ---');
representative_grid_indices = [];
for k = 1:size(representative_pixels, 1)
    rep_lon = representative_pixels(k, 1);
    rep_lat = representative_pixels(k, 2);
    [~, idx_lon] = min(abs(lon_grid - rep_lon));
    [~, idx_lat] = min(abs(lat_grid - rep_lat));
    
    if idx_lat >= 1 && idx_lat <= n_lat && idx_lon >= 1 && idx_lon <= n_lon
        representative_grid_indices = [representative_grid_indices; idx_lat, idx_lon];
    end
end
fprintf('Preparation complete. Total %d representative pixel grids.\n', size(representative_grid_indices, 1));

%% ========================================================================
% 6. Main Loop: Grid-by-Grid Comprehensive Analysis
% =========================================================================
disp('--- 6. Starting Grid-by-Grid Calculation (Using C-Vine and Modified Iteration Step) ---');

if USE_PARALLEL
    fprintf('Starting parallel pool...\n');
    try
        if isempty(gcp('nocreate'))
            parpool('Processes', min(MAX_PARALLEL_WORKERS, feature('numcores')));
        end
        pool = gcp('nocreate');
        fprintf('Connected to parallel pool with %d workers\n', pool.NumWorkers);
        parfor_option = 'parfor';
    catch ME
        fprintf('Failed to start parallel pool: %s\nSwitching to serial calculation.\n', ME.message);
        parfor_option = 'for';
    end
else
    fprintf('Using serial calculation.\n');
    parfor_option = 'for';
end

total_grids = n_lat * n_lon;
start_time = tic;

if strcmp(parfor_option, 'parfor')
    parfor i = 1:n_lat
        local_all_data = all_data;
        local_lon_grid = lon_grid;
        local_lat_grid = lat_grid;
        local_processed = 0;
        
        if SHOW_PROGRESS
            fprintf('Worker starting processing Latitude %d / %d\n', i, n_lat);
        end
        
        for j = 1:n_lon
            local_processed = local_processed + 1;
            if SHOW_PROGRESS && mod(local_processed, PROGRESS_INTERVAL) == 0
                fprintf('Latitude %d: Processed %d/%d grids\n', i, local_processed, n_lon);
            end
            
            idx = local_all_data.Latitude >= local_lat_grid(i) - GRID_RESOLUTION/2 & ...
                  local_all_data.Latitude < local_lat_grid(i) + GRID_RESOLUTION/2 & ...
                  local_all_data.Longitude >= local_lon_grid(j) - GRID_RESOLUTION/2 & ...
                  local_all_data.Longitude < local_lon_grid(j) + GRID_RESOLUTION/2;
            
            if sum(idx) < MIN_DATA_POINTS
                pit_data_grid{i,j} = struct();
                continue;
            end
            
            GPP_local = local_all_data.GPP(idx);
            SM_local = local_all_data.SM(idx);
            VPD_local = local_all_data.VPD(idx);
            
            current_pit_data = struct();
            
            try
                % --- 6.1 Marginal Distribution Selection (Based on AIC) ---
                [pd_sm, best_dist_sm, selection_table_sm] = select_best_dist(SM_local, marginal_dists);
                [pd_vpd, best_dist_vpd, selection_table_vpd] = select_best_dist(VPD_local, marginal_dists);
                [pd_gpp, best_dist_gpp, selection_table_gpp] = select_best_dist(GPP_local, marginal_dists);
                
                if isempty(pd_sm) || isempty(pd_vpd) || isempty(pd_gpp)
                    pit_data_grid{i,j} = struct();
                    continue;
                end
                
                best_dist_sm_grid{i,j} = best_dist_sm;
                best_dist_vpd_grid{i,j} = best_dist_vpd;
                best_dist_gpp_grid{i,j} = best_dist_gpp;
                
                dist_selection_results_sm_grid{i,j} = selection_table_sm;
                dist_selection_results_vpd_grid{i,j} = selection_table_vpd;
                dist_selection_results_gpp_grid{i,j} = selection_table_gpp;
                
                u_sm = max(min(cdf(pd_sm, SM_local), 1-1e-5), 1e-5);
                u_vpd = max(min(cdf(pd_vpd, VPD_local), 1-1e-5), 1e-5);
                u_gpp = max(min(cdf(pd_gpp, GPP_local), 1-1e-5), 1e-5);
                
                % --- 6.2 (SM-GPP) 2D Analysis (Iterative Step) ---
                data_sm_gpp = [u_sm, u_gpp];
                [best_family_sm_gpp, best_param_sm_gpp, best_aic_sm_gpp] = select_best_copula(data_sm_gpp, copula_families);
                
                best_copula_sm_gpp_grid{i,j} = best_family_sm_gpp;
                best_copula_aic_sm_gpp_grid(i,j) = best_aic_sm_gpp;
                
                [lower_tail_sm_gpp_grid(i,j), upper_tail_sm_gpp_grid(i,j)] = calculate_tail_dependence(best_family_sm_gpp, best_param_sm_gpp);
                
                u_gpp_star = cdf(pd_gpp, quantile(GPP_local, GPP_PERCENTILE/100));
                
                % [SM] Iteration step 0.01 m3/m3, max to min (Consistent with paper)
                sm_values = max(SM_local) : -0.01 : min(SM_local);
                if isempty(sm_values), sm_values = min(SM_local); end
                
                cond_prob_sm = arrayfun(@(sm_val) calc_psm_prob(sm_val, u_gpp_star, pd_sm, best_family_sm_gpp, best_param_sm_gpp), sm_values);
                PSM_grid(i,j) = max(cond_prob_sm);
                
                try
                    c_u2_u1_smgpp = copulacdf(best_family_sm_gpp, data_sm_gpp, best_param_sm_gpp, 'Conditional', 2);
                    uDist = makedist('Uniform',0,1);
                    
                    N_SM_GPP_grid(i,j) = sum(~isnan(c_u2_u1_smgpp));
                    Mean_SM_GPP_grid(i,j) = mean(c_u2_u1_smgpp,'omitnan');
                    Std_SM_GPP_grid(i,j) = std(c_u2_u1_smgpp,'omitnan');
                    
                    if N_SM_GPP_grid(i,j) > 0
                        [~, KS_p_SM_GPP_grid(i,j)] = kstest(c_u2_u1_smgpp, 'CDF', uDist);
                    else
                        KS_p_SM_GPP_grid(i,j) = NaN;
                    end
                catch
                end
                
                % --- 6.3 (VPD-GPP) 2D Analysis ---
                data_vpd_gpp = [u_vpd, u_gpp];
                [best_family_vpd_gpp, best_param_vpd_gpp, best_aic_vpd_gpp] = select_best_copula(data_vpd_gpp, copula_families);
                
                best_copula_vpd_gpp_grid{i,j} = best_family_vpd_gpp;
                best_copula_aic_vpd_gpp_grid(i,j) = best_aic_vpd_gpp;
                
                [lower_tail_vpd_gpp_grid(i,j), upper_tail_vpd_gpp_grid(i,j)] = calculate_tail_dependence(best_family_vpd_gpp, best_param_vpd_gpp);
                
                % [Modification] Iteration step 0.01 hPa, min to max (Consistent with paper)
                vpd_values = min(VPD_local) : 0.01 : max(VPD_local);
                if isempty(vpd_values), vpd_values = min(VPD_local); end
                
                cond_prob_vpd = arrayfun(@(vpd_val) calc_pvpd_prob(vpd_val, u_gpp_star, pd_vpd, best_family_vpd_gpp, best_param_vpd_gpp), vpd_values);
                PVPD_grid(i,j) = max(cond_prob_vpd);
                
                try
                    c_u2_u1_vpdgpp = copulacdf(best_family_vpd_gpp, data_vpd_gpp, best_param_vpd_gpp, 'Conditional', 2);
                    uDist = makedist('Uniform',0,1);
                    
                    N_VPD_GPP_grid(i,j) = sum(~isnan(c_u2_u1_vpdgpp));
                    Mean_VPD_GPP_grid(i,j) = mean(c_u2_u1_vpdgpp,'omitnan');
                    Std_VPD_GPP_grid(i,j) = std(c_u2_u1_vpdgpp,'omitnan');
                    
                    if N_VPD_GPP_grid(i,j) > 0
                        [~, KS_p_VPD_GPP_grid(i,j)] = kstest(c_u2_u1_vpdgpp, 'CDF', uDist);
                    else
                        KS_p_VPD_GPP_grid(i,j) = NaN;
                    end
                catch
                end
                
                % --- 6.4 3D Vine Copula Analysis (C-Vine Only) ---
                % Input data order fixed as: 1:SM, 2:GPP, 3:VPD
                U_data_3d_matrix = [u_sm, u_gpp, u_vpd];
                
                % Calculate C-Vine Structure
                try
                    [best_cvine_aic, best_cvine_bic, cvine_struct_full] = calc_cvine_structure_metrics(U_data_3d_matrix, copula_families);
                    c_vine_aic_grid(i,j) = best_cvine_aic;
                    c_vine_bic_grid(i,j) = best_cvine_bic;
                    % Store full structure for later calculation
                    best_vine_structure_grid{i,j} = cvine_struct_full;
                    
                    % --- 6.5 Calculate PSM&VPD using C-Vine ---
                    % Target: P(GPP<th | SM<th_sm, VPD>th_vpd)
                    % = [P(SM<th_sm, GPP<th, VPD>th_vpd)] / [P(SM<th_sm, VPD>th_vpd)]
                    % Numerator = P(SM<th_sm, GPP<th) - P(SM<th_sm, GPP<th, VPD<th_vpd)
                    % Denominator = P(SM<th_sm) - P(SM<th_sm, VPD<th_vpd)
                    
                    u_sm_10th = cdf(pd_sm, quantile(SM_local, SM_PERCENTILE/100));
                    u_vpd_90th = cdf(pd_vpd, quantile(VPD_local, VPD_PERCENTILE/100));
                    
                    % Use C-Vine integration to calculate 3D probability P(U1<u1, U2<u2, U3<u3)
                    % Variable Indices: 1=SM, 2=GPP, 3=VPD
                    
                    % Term 1: P(SM<th, GPP<th, VPD<1) -> Actually P(SM<th, GPP<th)
                    prob_sm_gpp_joint = copulacdf(best_family_sm_gpp, [u_sm_10th, u_gpp_star], best_param_sm_gpp);
                    num_term1 = prob_sm_gpp_joint;
                    
                    % Term 2: P(SM<th, GPP<th, VPD<th) -> 3D C-Vine Integration
                    num_term2 = calc_cvine_prob(cvine_struct_full, [u_sm_10th, u_gpp_star, u_vpd_90th]);
                    
                    numerator = num_term1 - num_term2;
                    
                    % Denominator Term: P(SM<th) - P(SM<th, VPD<th)
                    % Need Joint SM and VPD. If no direct SM-VPD Copula, marginalize GPP using C-Vine (set u_gpp=1)
                    den_term2 = calc_cvine_prob(cvine_struct_full, [u_sm_10th, 1, u_vpd_90th]);
                    
                    denominator = u_sm_10th - den_term2;
                    
                    if denominator > 1e-6
                        PSM_VPD_grid(i,j) = max(min(numerator / denominator, 1), 0);
                    else
                        PSM_VPD_grid(i,j) = 0;
                    end
                catch
                    c_vine_aic_grid(i,j) = NaN;
                end
                
                % --- 6.6 Generate PIT Data for Representative Points ---
                for k = 1:size(representative_pixels, 1)
                    if abs(representative_pixels(k, 1) - lon_grid(j)) <= GRID_RESOLUTION/2 && ...
                            abs(representative_pixels(k, 2) - local_lat_grid(i)) <= GRID_RESOLUTION/2
                        
                        pixel_name = sprintf('lon%d_lat%d', round(lon_grid(j)*10), round(lat_grid(i)*10));
                        current_pit_data.pixel_name = pixel_name;
                        current_pit_data.lon = lon_grid(j);
                        current_pit_data.lat = lat_grid(i);
                        
                        is_sm_gpp_pixel = any(ismember(sm_gpp_pixels, [lon_grid(j), lat_grid(i)], 'rows'));
                        is_vpd_gpp_pixel = any(ismember(vpd_gpp_pixels, [lon_grid(j), lat_grid(i)], 'rows'));
                        
                        if is_sm_gpp_pixel
                            current_pit_data.pixel_type = 'SM-GPP';
                        elseif is_vpd_gpp_pixel
                            current_pit_data.pixel_type = 'VPD-GPP';
                        else
                            current_pit_data.pixel_type = 'Unknown';
                        end
                        
                        try
                            c_u2_u1_smgpp = copulacdf(best_family_sm_gpp, data_sm_gpp, best_param_sm_gpp, 'Conditional', 2);
                            current_pit_data.sm_gpp = [data_sm_gpp(:,1), c_u2_u1_smgpp];
                        catch
                            current_pit_data.sm_gpp = [];
                        end
                        
                        try
                            c_u2_u1_vpdgpp = copulacdf(best_family_vpd_gpp, data_vpd_gpp, best_param_vpd_gpp, 'Conditional', 2);
                            current_pit_data.vpd_gpp = [data_vpd_gpp(:,1), c_u2_u1_vpdgpp];
                        catch
                            current_pit_data.vpd_gpp = [];
                        end
                        current_pit_data.vine_U_data = [u_sm, u_gpp, u_vpd];
                        break;
                    end
                end
                
            catch ME
                fprintf('Grid (lat=%d, lon=%d) calculation failed: %s\n', i, j, ME.message);
            end
            pit_data_grid{i,j} = current_pit_data;
        end
    end
end

total_time = toc(start_time);
fprintf('All grids calculation complete! Total time: %.1f minutes\n', total_time/60);

%% ========================================================================
% 7. Reassemble PIT Data into Original Structure Format
% =========================================================================
disp('--- 7. Reassembling PIT Data ---');
pit_data = struct();
for i = 1:n_lat
    for j = 1:n_lon
        current_data = pit_data_grid{i,j};
        if isstruct(current_data) && ~isempty(fieldnames(current_data))
            if isfield(current_data, 'pixel_name') && ~isempty(current_data.pixel_name)
                pixel_name = current_data.pixel_name;
                valid_field_name = matlab.lang.makeValidName(pixel_name);
                pit_data.(valid_field_name).sm_gpp = current_data.sm_gpp;
                pit_data.(valid_field_name).vpd_gpp = current_data.vpd_gpp;
                pit_data.(valid_field_name).vine_U_data = current_data.vine_U_data;
                pit_data.(valid_field_name).lon = current_data.lon;
                pit_data.(valid_field_name).lat = current_data.lat;
                pit_data.(valid_field_name).pixel_type = current_data.pixel_type;
            end
        end
    end
end
clear pit_data_grid current_pit_data;

%% ========================================================================
% 8. Save All Results
% =========================================================================
disp('--- 8. Saving Results to .mat File ---');
save(fullfile(OUTPUT_DIR, 'unified_analysis_results_group1.mat'), ...
    'lon_grid', 'lat_grid', 'PSM_grid', 'PVPD_grid', 'PSM_VPD_grid', ...
    'best_dist_sm_grid', 'best_dist_vpd_grid', 'best_dist_gpp_grid', ...
    'dist_selection_results_sm_grid', 'dist_selection_results_vpd_grid', 'dist_selection_results_gpp_grid', ...
    'best_copula_sm_gpp_grid', 'best_copula_vpd_gpp_grid', ...
    'best_copula_aic_sm_gpp_grid','best_copula_aic_vpd_gpp_grid', ...
    'lower_tail_sm_gpp_grid', 'upper_tail_sm_gpp_grid', ...
    'lower_tail_vpd_gpp_grid', 'upper_tail_vpd_gpp_grid', ...
    'best_vine_structure_grid', ...
    'c_vine_aic_grid', 'c_vine_bic_grid', ... % Saving only C-Vine AIC/BIC
    'N_SM_GPP_grid','Mean_SM_GPP_grid','Std_SM_GPP_grid','KS_p_SM_GPP_grid', ...
    'N_VPD_GPP_grid','Mean_VPD_GPP_grid','Std_VPD_GPP_grid','KS_p_VPD_GPP_grid',...
    'pit_data', 'sm_gpp_pixels', 'vpd_gpp_pixels', ...
    '-v7.3');
disp(['Results saved to: ', fullfile(OUTPUT_DIR, 'unified_analysis_results_group1.mat')]);

%% ========================================================================
% 9. Summary Statistics Output (Completing Missing Parts)
% =========================================================================
disp('--- 9. Summarizing Marginal Distribution Statistics ---');
% Extract SM Best Distributions
sm_dists_list = best_dist_sm_grid(:);
sm_dists_list = sm_dists_list(~cellfun('isempty', sm_dists_list));
unique_sm_dists = unique(sm_dists_list);
sm_counts = zeros(size(unique_sm_dists));
for k = 1:length(unique_sm_dists), sm_counts(k) = sum(strcmp(sm_dists_list, unique_sm_dists{k})); end
sm_percent = sm_counts / sum(sm_counts) * 100;

% Extract VPD Best Distributions
vpd_dists_list = best_dist_vpd_grid(:);
vpd_dists_list = vpd_dists_list(~cellfun('isempty', vpd_dists_list));
unique_vpd_dists = unique(vpd_dists_list);
vpd_counts = zeros(size(unique_vpd_dists));
for k = 1:length(unique_vpd_dists), vpd_counts(k) = sum(strcmp(vpd_dists_list, unique_vpd_dists{k})); end
vpd_percent = vpd_counts / sum(vpd_counts) * 100;

% Extract GPP Best Distributions
gpp_dists_list = best_dist_gpp_grid(:);
gpp_dists_list = gpp_dists_list(~cellfun('isempty', gpp_dists_list));
unique_gpp_dists = unique(gpp_dists_list);
gpp_counts = zeros(size(unique_gpp_dists));
for k = 1:length(unique_gpp_dists), gpp_counts(k) = sum(strcmp(gpp_dists_list, unique_gpp_dists{k})); end
gpp_percent = gpp_counts / sum(gpp_counts) * 100;

fprintf('Marginal distribution statistics complete.\n');

%% ========================================================================
% 10. Plotting Section
% =========================================================================
disp('--- 10. Starting Plotting ---');

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
    disp('Parallel pool closed to avoid graphics system conflicts');
end
close all;

try
    % 10.1 Original Maps
    
    % --- 1. PSM Map Modification ---
    % Create White to Dark Blue Gradient (0=White, 1=Blue)
    cmap_blue = [linspace(1,0,64)' linspace(1,0,64)' linspace(1,1,64)']; % Blue: R0 G0 B1
    
    figure('Name', 'PSM', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
    % Pass the custom colormap
    plot_china_map_safe(lon_grid, lat_grid, PSM_grid, 'PSM', 'Maximum triggering probability of SM(%)', cmap_blue);
    print(gcf, fullfile(OUTPUT_DIR, 'PSM_Map.png'), '-dpng', '-r300');
    
    % --- 2. PVPD Map Modification ---
    % Create White to Bright Red Gradient (0=White, 1=Red)
    cmap_red = [ones(64,1) linspace(1,0,64)' linspace(1,0,64)']; % Red: R1 G0 B0
    
    figure('Name', 'PVPD', 'NumberTitle', 'off', 'Position', [150, 150, 800, 600]);
    % Pass the custom colormap
    plot_china_map_safe(lon_grid, lat_grid, PVPD_grid, 'PVPD', 'Maximum triggering probability of VPD(%)', cmap_red);
    print(gcf, fullfile(OUTPUT_DIR, 'PVPD_Map.png'), '-dpng', '-r300');
    
    % --- 3. PSM&VPD Map Modification ---
    % Keep Purple Gradient but apply same font/tick settings
    cmap_purple = [linspace(1,0.5,64)', linspace(1,0,64)', linspace(1,0.5,64)'];
    
    figure('Name', 'PSM&VPD', 'NumberTitle', 'off', 'Position', [200, 200, 800, 600]);
    plot_china_map_safe(lon_grid, lat_grid, PSM_VPD_grid, 'PSM&VPD', 'Trigger probability under low SM & high VPD (%)', cmap_purple);
    print(gcf, fullfile(OUTPUT_DIR, 'PSM_VPD_Map.png'), '-dpng', '-r300');
    
    % 10.2 Supplementary Material (Best Copula SM-GPP)
    copula_map_sm_gpp = zeros(n_lat, n_lon);
    for i = 1:numel(copula_families)
        idx = strcmp(best_copula_sm_gpp_grid, copula_families{i});
        copula_map_sm_gpp(idx) = i;
    end
    figure('Name', 'Best Copula Family (SM-GPP)', 'NumberTitle', 'off', 'Position', [250, 250, 800, 600]);
    plot_china_map_categorical_safe(lon_grid, lat_grid, copula_map_sm_gpp, 'Best Copula Family (SM-GPP)', copula_families);
    print(gcf, fullfile(OUTPUT_DIR, 'Best_Copula_SM_GPP.png'), '-dpng', '-r300');
    
    % 10.3 Tail Dependence
    figure('Name', 'SM-GPP Tail Dependence', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
    subplot(2, 2, 1);
    plot_china_map_simple(lon_grid, lat_grid, lower_tail_sm_gpp_grid, 'Lower Tail Dependence (SM-GPP)', 'jet', [0, 1]);
    title('(a) Lower Tail Dependence (SM-GPP)', 'FontSize', 12, 'FontName', 'Times New Roman');
    
    subplot(2, 2, 2);
    plot_china_map_simple(lon_grid, lat_grid, upper_tail_sm_gpp_grid, 'Upper Tail Dependence (SM-GPP)', 'jet', [0, 1]);
    title('(b) Upper Tail Dependence (SM-GPP)', 'FontSize', 12, 'FontName', 'Times New Roman');
    
    subplot(2, 2, 3);
    plot_china_map_simple(lon_grid, lat_grid, lower_tail_vpd_gpp_grid, 'Lower Tail Dependence (VPD-GPP)', 'jet', [0, 1]);
    title('(c) Lower Tail Dependence (VPD-GPP)', 'FontSize', 12, 'FontName', 'Times New Roman');
    
    subplot(2, 2, 4);
    plot_china_map_simple(lon_grid, lat_grid, upper_tail_vpd_gpp_grid, 'Upper Tail Dependence (VPD-GPP)', 'jet', [0, 1]);
    title('(d) Upper Tail Dependence (VPD-GPP)', 'FontSize', 12, 'FontName', 'Times New Roman');
    
    sgtitle('Tail Dependence Coefficients', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    print(gcf, fullfile(OUTPUT_DIR, 'Tail_Dependence.png'), '-dpng', '-r300');
    
    % 10.4 Vine Structure Selection (Showing C-Vine Root)
    % Extract Root Node info for plotting
    root_map = zeros(n_lat, n_lon);
    for i=1:n_lat
        for j=1:n_lon
            if ~isempty(best_vine_structure_grid{i,j})
                % order(1) is the root
                r_idx = best_vine_structure_grid{i,j}.order(1);
                root_map(i,j) = r_idx;
            end
        end
    end
    figure('Name', 'C-Vine Root Variable', 'NumberTitle', 'off', 'Position', [300, 300, 800, 600]);
    plot_china_map_categorical_safe(lon_grid, lat_grid, root_map, 'Best C-Vine Root Variable', {'SM', 'GPP', 'VPD'});
    print(gcf, fullfile(OUTPUT_DIR, 'Best_Vine_Structure.png'), '-dpng', '-r300');
    
    % 10.5 Display PIT Test Figures
    pixel_names = fieldnames(pit_data);
    if ~isempty(pixel_names)
        num_pixels = numel(pixel_names);
        if num_pixels <= 4, nrows = 2; ncols = 2;
        elseif num_pixels <= 6, nrows = 2; ncols = 3;
        elseif num_pixels <= 9, nrows = 3; ncols = 3;
        else, nrows = ceil(sqrt(num_pixels)); ncols = ceil(num_pixels / nrows);
        end
        
        fig_pit = figure('Name', 'PIT Test Figures', 'NumberTitle', 'off', 'Position', [100, 100, ncols*400, nrows*300]);
        x_limits = [0, 1];
        
        for i = 1:num_pixels
            subplot(nrows, ncols, i);
            set(gca, 'FontName', 'Times New Roman'); % Set axis font
            
            if isfield(pit_data.(pixel_names{i}), 'sm_gpp') && ~isempty(pit_data.(pixel_names{i}).sm_gpp)
                pit_values = pit_data.(pixel_names{i}).sm_gpp;
                histogram(pit_values(:,2), 'Normalization', 'probability', ...
                    'NumBins', 10, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
                hold on;
                
                uniform_prob = 1/10;
                x_range = xlim;
                y_range = ylim;
                line(x_limits, [uniform_prob, uniform_prob], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
                
                lon_str = sprintf('%.1f°E', pit_data.(pixel_names{i}).lon);
                lat_str = sprintf('%.1f°N', pit_data.(pixel_names{i}).lat);
                
                if isfield(pit_data.(pixel_names{i}), 'pixel_type')
                    pixel_type_str = pit_data.(pixel_names{i}).pixel_type;
                else
                    pixel_type_str = 'Unknown';
                end
                
                title(sprintf('(%d) Lon: %s, Lat: %s\nType: %s', i, lon_str, lat_str, pixel_type_str), ...
                    'FontSize', 9, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
                xlabel('PIT values', 'FontSize', 9, 'FontName', 'Times New Roman');
                ylabel('Probability', 'FontSize', 9, 'FontName', 'Times New Roman');
                xlim(x_limits);
                grid on; box on;
            else
                text(0.5, 0.5, 'No PIT data', 'HorizontalAlignment', 'center', 'FontName', 'Times New Roman');
            end
        end
        set(gcf, 'Color', 'w');
        sgtitle('PIT Goodness-of-Fit Test for Representative Pixels', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        print(fig_pit, fullfile(OUTPUT_DIR, 'PIT_test_figure.png'), '-dpng', '-r300');
    end
    
    % 10.6 Marginal Distribution Statistics Plot
    % Variables calculated in Section 9
    if exist('sm_percent', 'var') && ~isempty(sm_percent)
        figure('Name', 'Marginal Distribution Statistics', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);
        
        subplot(1, 3, 1);
        set(gca, 'FontName', 'Times New Roman'); % Set font
        bar(1:length(sm_percent), sm_percent, 'FaceColor', [0.2, 0.4, 0.8]);
        xticks(1:length(sm_percent));
        xticklabels(unique_sm_dists);
        xtickangle(45);
        ylabel('Percentage (%)', 'FontName', 'Times New Roman');
        title('(a) SM Marginal Distribution', 'FontName', 'Times New Roman');
        grid on;
        
        subplot(1, 3, 2);
        set(gca, 'FontName', 'Times New Roman'); % Set font
        bar(1:length(vpd_percent), vpd_percent, 'FaceColor', [0.8, 0.2, 0.2]);
        xticks(1:length(vpd_percent));
        xticklabels(unique_vpd_dists);
        xtickangle(45);
        ylabel('Percentage (%)', 'FontName', 'Times New Roman');
        title('(b) VPD Marginal Distribution', 'FontName', 'Times New Roman');
        grid on;
        
        subplot(1, 3, 3);
        set(gca, 'FontName', 'Times New Roman'); % Set font
        bar(1:length(gpp_percent), gpp_percent, 'FaceColor', [0.2, 0.8, 0.4]);
        xticks(1:length(gpp_percent));
        xticklabels(unique_gpp_dists);
        xtickangle(45);
        ylabel('Percentage (%)', 'FontName', 'Times New Roman');
        title('(c) GPP Marginal Distribution', 'FontName', 'Times New Roman');
        grid on;
        
        sgtitle('Marginal Distribution Selection Statistics', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        print(gcf, fullfile(OUTPUT_DIR, 'Marginal_Distribution_Stats.png'), '-dpng', '-r300');
    else
        warning('Statistical variables are empty, cannot plot marginal distribution statistics');
    end
    
    % 10.7 Plot PIT Histograms for Specified Pixels
    create_specified_pixel_pit_plots(OUTPUT_DIR, pit_data, sm_gpp_pixels, vpd_gpp_pixels, PIT_GOOD_FIT_THRESHOLD);
    
    disp('--- All plotting complete ---');
    
catch ME
    warning('Error during plotting: %s\nHowever, calculations are complete and results saved.', ME.message);
end

%% ========================================================================
% 11. Export Five Tables (CSV)
% =========================================================================
disp('--- 11. Exporting Table Data ---');

% ---------- Table 1: Marginal Distribution Comparison (With KS/AIC, BIC removed) ----------
T1 = table();
rowCounter = 0;
for i = 1:n_lat
    for j = 1:n_lon
        if isempty(dist_selection_results_sm_grid{i,j}), continue; end
        Tsm = expand_one_var(dist_selection_results_sm_grid{i,j}, 'SM', best_dist_sm_grid{i,j}, lat_grid(i), lon_grid(j));
        Tvpd = expand_one_var(dist_selection_results_vpd_grid{i,j}, 'VPD', best_dist_vpd_grid{i,j}, lat_grid(i), lon_grid(j));
        Tgpp = expand_one_var(dist_selection_results_gpp_grid{i,j}, 'GPP', best_dist_gpp_grid{i,j}, lat_grid(i), lon_grid(j));
        
        Tall = [Tsm; Tvpd; Tgpp];
        if ~isempty(Tall)
            if rowCounter==0, T1 = Tall; else, T1 = [T1; Tall]; end
            rowCounter = rowCounter + height(Tall);
        end
    end
end
% Remove BIC Column
if ismember('BIC', T1.Properties.VariableNames)
    T1 = removevars(T1, 'BIC');
end
t1_path = fullfile(OUTPUT_DIR, 'table1_marginal_selection.csv');
if ~isempty(T1), writetable(T1, t1_path); end

% ---------- Table 2: 2D Copula Best Family (With AIC) ----------
lat_col = []; lon_col = []; pair_col = {}; family_col = {}; aic_col = [];
for i = 1:n_lat
    for j = 1:n_lon
        if ~isempty(best_copula_sm_gpp_grid{i,j})
            lat_col(end+1,1) = lat_grid(i);
            lon_col(end+1,1) = lon_grid(j);
            pair_col{end+1,1} = 'SM-GPP';
            family_col{end+1,1} = best_copula_sm_gpp_grid{i,j};
            aic_col(end+1,1) = best_copula_aic_sm_gpp_grid(i,j);
        end
        if ~isempty(best_copula_vpd_gpp_grid{i,j})
            lat_col(end+1,1) = lat_grid(i);
            lon_col(end+1,1) = lon_grid(j);
            pair_col{end+1,1} = 'VPD-GPP';
            family_col{end+1,1} = best_copula_vpd_gpp_grid{i,j};
            aic_col(end+1,1) = best_copula_aic_vpd_gpp_grid(i,j);
        end
    end
end
T2 = table(lat_col, lon_col, pair_col, family_col, aic_col, 'VariableNames', {'Latitude','Longitude','Pair','BestFamily','AIC'});
t2_path = fullfile(OUTPUT_DIR, 'table2_best_copula_2d.csv');
if ~isempty(T2), writetable(T2, t2_path); end

% ---------- Table 3: Tail Dependence Coefficients ----------
[LONm, LATm] = meshgrid(lon_grid, lat_grid);
T3 = table();
T3.Latitude = LATm(:);
T3.Longitude = LONm(:);
T3.LambdaL_SM_GPP = lower_tail_sm_gpp_grid(:);
T3.LambdaU_SM_GPP = upper_tail_sm_gpp_grid(:);
T3.LambdaL_VPD_GPP = lower_tail_vpd_gpp_grid(:);
T3.LambdaU_VPD_GPP = upper_tail_vpd_gpp_grid(:);

valid_rows = ~(isnan(T3.LambdaL_SM_GPP) & isnan(T3.LambdaU_SM_GPP));
T3 = T3(valid_rows, :);
t3_path = fullfile(OUTPUT_DIR, 'table3_tail_dependence.csv');
if ~isempty(T3), writetable(T3, t3_path); end

% ---------- Table 4: C-Vine Copula Structure & AIC/BIC ----------
T4 = table();
T4.Latitude = LATm(:);
T4.Longitude = LONm(:);
T4.C_Vine_AIC = c_vine_aic_grid(:);
T4.C_Vine_BIC = c_vine_bic_grid(:);
% D-Vine columns removed
valid_rows4 = ~isnan(T4.C_Vine_AIC);
T4 = T4(valid_rows4, :);
t4_path = fullfile(OUTPUT_DIR, 'table4_vine_cvine_results.csv');
if ~isempty(T4), writetable(T4, t4_path); end

% ---------- Table 5: PIT Statistics ----------
P5 = table();
P5.Latitude = LATm(:); P5.Longitude = LONm(:);
P5.N_SM_GPP = N_SM_GPP_grid(:);
P5.Mean_SM_GPP = Mean_SM_GPP_grid(:);
P5.Std_SM_GPP = Std_SM_GPP_grid(:);
P5.KS_p_SM_GPP = KS_p_SM_GPP_grid(:);

P5.N_VPD_GPP = N_VPD_GPP_grid(:);
P5.Mean_VPD_GPP = Mean_VPD_GPP_grid(:);
P5.Std_VPD_GPP = Std_VPD_GPP_grid(:);
P5.KS_p_VPD_GPP = KS_p_VPD_GPP_grid(:);

valid_p5 = ~( (isnan(P5.N_SM_GPP) & isnan(P5.N_VPD_GPP)) | (P5.N_SM_GPP==0 & P5.N_VPD_GPP==0) );
P5 = P5(valid_p5, :);
t5_path = fullfile(OUTPUT_DIR, 'table5_pit_stats.csv');
if ~isempty(P5), writetable(P5, t5_path); end

% ---------- Table 6: Good Pixels List ----------
T6 = create_pit_good_pixels_table(lon_grid, lat_grid, KS_p_SM_GPP_grid, KS_p_VPD_GPP_grid, N_SM_GPP_grid, N_VPD_GPP_grid, Mean_SM_GPP_grid, Mean_VPD_GPP_grid, PIT_GOOD_FIT_THRESHOLD);
if ~isempty(T6)
    t6_path = fullfile(OUTPUT_DIR, 'table6_pit_good_pixels.csv');
    writetable(T6, t6_path);
end
fprintf('All table exports complete.\n');

% ========================================================================
% Helper Functions
% ========================================================================

function Tvar = expand_one_var(sel_table, varname, best_name, latv, lonv)
    if isempty(sel_table), Tvar = table(); return; end
    Tvar = sel_table;
    Tvar.Variable = repmat({varname}, height(Tvar), 1);
    Tvar.Latitude = repmat(latv, height(Tvar), 1);
    Tvar.Longitude = repmat(lonv, height(Tvar), 1);
    Tvar.IsBest = strcmp(Tvar.Distribution, best_name);
    
    % Ensure columns included
    cols = {'Latitude','Longitude','Variable','Distribution','KS_p_value','KS_Pass','AIC','IsBest'};
    if ismember('BIC', Tvar.Properties.VariableNames), cols = [cols, 'BIC']; end
    Tvar = Tvar(:, cols);
end

function [pd, best_dist_name, results_table] = select_best_dist(data, candidates)
    % Select using AIC only, KS must pass
    num_candidates = length(candidates);
    results = cell(num_candidates, 5);
    
    for i = 1:num_candidates
        dist_name = candidates{i};
        results{i, 1} = dist_name;
        
        try
            current_pd = fitdist(data, dist_name);
            if strcmpi(dist_name, 'GeneralizedExtremeValue'), k = 3; else, k = 2; end
            
            n = length(data);
            try, logL = sum(log(pdf(current_pd, data))); catch, logL = -inf; end
            
            results{i, 2} = 2*k - 2*logL; % AIC
            results{i, 3} = k*log(n) - 2*logL; % BIC (Calculated for record, not used for selection)
            
            [h, p_value] = kstest(data, 'CDF', current_pd);
            results{i, 4} = p_value;
            results{i, 5} = ~h; % KS Pass
        catch
            results{i, 2} = inf;
            results{i, 3} = inf;
            results{i, 4} = 0;
            results{i, 5} = false;
        end
    end
    
    results_table = cell2table(results, 'VariableNames', {'Distribution', 'AIC', 'BIC', 'KS_p_value', 'KS_Pass'});
    pd = [];
    best_dist_name = '';
    
    % Filter candidates that passed KS test
    passed_candidates = results_table(results_table.KS_Pass == true, :);
    
    if ~isempty(passed_candidates)
        % Select minimum AIC
        [~, min_idx] = min(passed_candidates.AIC);
        best_dist_name = passed_candidates.Distribution{min_idx};
        try, pd = fitdist(data, best_dist_name); catch, pd = []; best_dist_name = ''; end
    else
        % If none passed KS, select min AIC as fallback
        [~, min_idx] = min(results_table.AIC);
        if ~isinf(results_table.AIC(min_idx))
            best_dist_name = results_table.Distribution{min_idx};
            try, pd = fitdist(data, best_dist_name); catch, pd = []; best_dist_name = ''; end
        end
    end
    if isempty(pd), best_dist_name = 'None'; end
end

function [best_family, best_param, best_aic] = select_best_copula(u_data, candidates)
    best_aic = inf;
    best_family = '';
    best_param = [];
    
    for i = 1:length(candidates)
        family = candidates{i};
        try
            param = copulafit(family, u_data);
            logL = sum(log(copulapdf(family, u_data, param)));
            aic = -2 * logL + 2 * length(param);
            
            if aic < best_aic, best_aic = aic; best_family = family; best_param = param; end
        catch, continue; end
    end
end

function [lambda_l, lambda_u] = calculate_tail_dependence(family, param)
    lambda_l = 0;
    lambda_u = 0;
    switch lower(family)
        case 'clayton', lambda_l = 2^(-1/param);
        case 'gumbel', lambda_u = 2 - 2^(1/param);
        case 't'
            rho = param(1); nu = param(2);
            lambda_l = 2 * tcdf(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1);
            lambda_u = lambda_l;
    end
end

function prob = calc_psm_prob(sm_val, u_gpp_star, pd_sm, family, param)
    u_sm_star = cdf(pd_sm, sm_val);
    if u_sm_star > 1e-6
        prob = copulacdf(family, [u_sm_star, u_gpp_star], param) / u_sm_star;
    else
        prob = 0;
    end
end

function prob = calc_pvpd_prob(vpd_val, u_gpp_star, pd_vpd, family, param)
    u_vpd_star = cdf(pd_vpd, vpd_val);
    if (1 - u_vpd_star) > 1e-6
        prob = (u_gpp_star - copulacdf(family, [u_vpd_star, u_gpp_star], param)) / (1 - u_vpd_star);
    else
        prob = 0;
    end
end

function T = create_pit_good_pixels_table(lon_grid, lat_grid, KS_p_SM, KS_p_VPD, N_SM, N_VPD, Mean_SM, Mean_VPD, threshold)
    lat_col = []; lon_col = []; type_col = {}; ks_p_col = []; n_col = []; mean_col = [];
    [n_lat, n_lon] = size(KS_p_SM);
    
    for i = 1:n_lat
        for j = 1:n_lon
            if ~isnan(KS_p_SM(i,j)) && KS_p_SM(i,j) > threshold
                lat_col = [lat_col; lat_grid(i)];
                lon_col = [lon_col; lon_grid(j)];
                type_col = [type_col; {'SM-GPP'}];
                ks_p_col = [ks_p_col; KS_p_SM(i,j)];
                n_col = [n_col; N_SM(i,j)];
                mean_col = [mean_col; Mean_SM(i,j)];
            end
            if ~isnan(KS_p_VPD(i,j)) && KS_p_VPD(i,j) > threshold
                lat_col = [lat_col; lat_grid(i)];
                lon_col = [lon_col; lon_grid(j)];
                type_col = [type_col; {'VPD-GPP'}];
                ks_p_col = [ks_p_col; KS_p_VPD(i,j)];
                n_col = [n_col; N_VPD(i,j)];
                mean_col = [mean_col; Mean_VPD(i,j)];
            end
        end
    end
    T = table(lat_col, lon_col, type_col, ks_p_col, n_col, mean_col, 'VariableNames', {'Latitude', 'Longitude', 'Type', 'KS_p_value', 'Sample_Size', 'Mean_PIT'});
end

% --- C-Vine Specific Calculation Functions ---

% 1. Calculate C-Vine Structure Metrics
function [best_aic, best_bic, best_struct] = calc_cvine_structure_metrics(U, candidates)
    % U: [SM, GPP, VPD] (1, 2, 3)
    % C-Vine Root Node Traversal
    n = size(U, 1);
    best_aic = inf;
    best_bic = inf;
    best_struct = [];
    vars = 1:3;
    
    % C-Vine Feature: One root connects to two others
    for root = 1:3
        leaves = setdiff(vars, root);
        leaf1 = leaves(1);
        leaf2 = leaves(2);
        
        current_aic = 0;
        current_k = 0;
        
        % Tree 1: (Root, Leaf1), (Root, Leaf2)
        % Pair 1
        u_pair1 = U(:, [root, leaf1]);
        [fam1, param1, aic1] = select_best_copula(u_pair1, candidates);
        current_aic = current_aic + aic1;
        current_k = current_k + numel(param1);
        
        % Pair 2
        u_pair2 = U(:, [root, leaf2]);
        [fam2, param2, aic2] = select_best_copula(u_pair2, candidates);
        current_aic = current_aic + aic2;
        current_k = current_k + numel(param2);
        
        % Tree 2 Input: Conditional CDFs
        % v1 = F(Leaf1 | Root), v2 = F(Leaf2 | Root)
        try
            v1 = copulacdf(fam1, [u_pair1(:,2), u_pair1(:,1)], param1, 'Conditional', 1); % F(u2|u1) where u1 is root
            v2 = copulacdf(fam2, [u_pair2(:,2), u_pair2(:,1)], param2, 'Conditional', 1);
        catch
            continue;
        end
        
        % Tree 2: (Leaf1, Leaf2 | Root)
        [fam3, param3, aic3] = select_best_copula([v1, v2], candidates);
        current_aic = current_aic + aic3;
        current_k = current_k + numel(param3);
        
        logL_total = (2*current_k - current_aic) / 2;
        current_bic = -2 * logL_total + current_k * log(n);
        
        if current_aic < best_aic
            best_aic = current_aic;
            best_bic = current_bic;
            % Record structure: order = [Root, Leaf1, Leaf2]
            best_struct.order = [root, leaf1, leaf2];
            best_struct.fam1 = fam1;
            best_struct.param1 = param1; % (Root, Leaf1)
            best_struct.fam2 = fam2;
            best_struct.param2 = param2; % (Root, Leaf2)
            best_struct.fam3 = fam3;
            best_struct.param3 = param3; % (Leaf1, Leaf2 | Root)
        end
    end
end

% 2. Calculate 3D Cumulative Probability F(u1, u2, u3) using C-Vine
% Formula: F(x,y,z) = integral_{0}^{u_root} C_{L1,L2|R} ( F(L1|w), F(L2|w) ) dw
function p = calc_cvine_prob(cv_struct, u_vec)
    % u_vec order corresponds to original variables [SM, GPP, VPD] (1,2,3)
    % cv_struct.order indicates indices of [Root, Leaf1, Leaf2]
    root_idx = cv_struct.order(1);
    l1_idx = cv_struct.order(2);
    l2_idx = cv_struct.order(3);
    
    u_root = u_vec(root_idx);
    u_l1 = u_vec(l1_idx);
    u_l2 = u_vec(l2_idx);
    
    % Define Integrand
    % w is probability value of root node (0 to u_root)
    % integrand = C_3 ( h(u_l1 | w), h(u_l2 | w) )
    integrand = @(w) cvine_integrand(w, u_l1, u_l2, cv_struct);
    
    % Numerical Integration
    try
        p = integral(integrand, 0, u_root, 'ArrayValued', true, 'AbsTol', 1e-4, 'RelTol', 1e-3);
    catch
        p = NaN;
    end
end

function val = cvine_integrand(w, u_l1, u_l2, s)
    % Calculate conditional probability h-functions
    % Tree 1 Pair 1: (Root, Leaf1) -> h1 = F(u_l1 | w)
    % Note: copulacdf(..., 'Conditional', 1) calculates F(u2|u1).
    % Here input is [w, u_l1] -> We want F(u_l1 | w), i.e., prob of 2nd var given 1st var
    % So call copulacdf(..., [w, u_l1], ..., 'Conditional', 1)
    
    % Handle vectorized input w
    n = length(w);
    
    % Avoid boundary errors if w is 0 or 1
    w(w<1e-6) = 1e-6;
    w(w>1-1e-6) = 1-1e-6;
    
    % Calculate h1 = F(Leaf1 | Root=w)
    try
        h1 = copulacdf(s.fam1, [w(:), repmat(u_l1,n,1)], s.param1, 'Conditional', 1);
        h2 = copulacdf(s.fam2, [w(:), repmat(u_l2,n,1)], s.param2, 'Conditional', 1);
        
        % Calculate Tree 2 Copula: C_3(h1, h2)
        val = copulacdf(s.fam3, [h1, h2], s.param3);
    catch
        val = zeros(n,1);
    end
end

% --- Plotting Helper Functions ---

function plot_china_map_safe(lon_grid, lat_grid, data_grid, fig_title, cbar_label, cmap_matrix)
    ax = gca;
    imagesc(lon_grid, lat_grid, data_grid, 'AlphaData', ~isnan(data_grid));
    set(ax, 'YDir', 'normal');
    set(ax, 'FontName', 'Times New Roman'); % Set axis font
    
    % Apply Custom Colormap
    colormap(ax, cmap_matrix);
    
    % Setup Colorbar
    h = colorbar;
    
    % Set colorbar ticks and labels as requested
    set(h, 'Ticks', 0:0.2:1, 'TickLabels', {'0','0.2','0.4','0.6','0.8','1'}, ...
        'FontName', 'Times New Roman', 'FontSize', 18);
    
    ylabel(h, cbar_label, 'FontSize', 18, 'FontName', 'Times New Roman');
    
    title(fig_title, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    xlabel('Longitude (°E)', 'FontName', 'Times New Roman');
    ylabel('Latitude (°N)', 'FontName', 'Times New Roman');
    xlim([70, 140]);
    ylim([15, 55]);
    grid on;
    box on;
end

function plot_china_map_categorical_safe(lon_grid, lat_grid, data_grid, fig_title, categories)
    ax = gca;
    imagesc(lon_grid, lat_grid, data_grid, 'AlphaData', data_grid > 0);
    set(ax, 'YDir', 'normal');
    set(ax, 'FontName', 'Times New Roman'); % Set axis font
    
    num_cats = length(categories);
    cmap = parula(num_cats);
    colormap(gca, cmap);
    caxis([0.5, num_cats + 0.5]);
    
    h = colorbar;
    set(h, 'Ticks', 1:num_cats, 'TickLabels', categories);
    set(h, 'FontName', 'Times New Roman'); % Set Colorbar font
    
    title(fig_title, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    xlabel('Longitude (°E)', 'FontName', 'Times New Roman');
    ylabel('Latitude (°N)', 'FontName', 'Times New Roman');
    xlim([70, 140]);
    ylim([15, 55]);
    grid on;
    box on;
end

function plot_china_map_simple(lon_grid, lat_grid, data_grid, fig_title, cmap_name, clims)
    ax = gca;
    imagesc(lon_grid, lat_grid, data_grid, 'AlphaData', ~isnan(data_grid));
    set(ax, 'YDir', 'normal');
    set(ax, 'FontName', 'Times New Roman'); % Set axis font
    
    colormap(gca, cmap_name);
    if nargin >= 6 && ~isempty(clims), caxis(clims); end
    
    h = colorbar;
    set(h, 'FontName', 'Times New Roman'); % Set Colorbar font
    
    title(fig_title, 'FontSize', 10, 'FontName', 'Times New Roman');
    xlabel('Lon', 'FontName', 'Times New Roman');
    ylabel('Lat', 'FontName', 'Times New Roman');
    xlim([70, 140]);
    ylim([15, 55]);
    grid on;
    box on;
end

function create_specified_pixel_pit_plots(output_dir, pit_data, sm_pixels, vpd_pixels, threshold)
    plot_group(output_dir, pit_data, sm_pixels, 'SM-GPP_Selected', 'SM-GPP');
    plot_group(output_dir, pit_data, vpd_pixels, 'VPD-GPP_Selected', 'VPD-GPP');
end

function plot_group(out_dir, p_data, pixel_list, filename_prefix, type_label)
    num_pix = size(pixel_list, 1);
    if num_pix == 0, return; end
    
    nrows = ceil(sqrt(num_pix));
    ncols = ceil(num_pix / nrows);
    
    f = figure('Name', [type_label ' Selected Pixels'], 'Visible', 'off', 'Position', [100, 100, ncols*300, nrows*250]);
    
    for k = 1:num_pix
        px_lon = pixel_list(k, 1);
        px_lat = pixel_list(k, 2);
        p_name = sprintf('lon%d_lat%d', round(px_lon*10), round(px_lat*10));
        valid_name = matlab.lang.makeValidName(p_name);
        
        subplot(nrows, ncols, k);
        set(gca, 'FontName', 'Times New Roman'); % Set axis font
        
        found = false;
        if isfield(p_data, valid_name)
            d = p_data.(valid_name);
            if contains(type_label, 'SM') && isfield(d, 'sm_gpp') && ~isempty(d.sm_gpp)
                vals = d.sm_gpp(:,2);
                found = true;
            elseif contains(type_label, 'VPD') && isfield(d, 'vpd_gpp') && ~isempty(d.vpd_gpp)
                vals = d.vpd_gpp(:,2);
                found = true;
            end
        end
        
        if found
            histogram(vals, 'Normalization', 'probability', 'NumBins', 10, 'FaceColor', [0.4 0.6 0.9]);
            hold on;
            yline(0.1, 'r--', 'LineWidth', 1.5);
            title(sprintf('Lon:%.1f Lat:%.1f', px_lon, px_lat), 'FontSize', 8, 'FontName', 'Times New Roman');
            xlim([0 1]);
        else
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'FontName', 'Times New Roman');
            title(sprintf('Lon:%.1f Lat:%.1f', px_lon, px_lat), 'FontSize', 8, 'FontName', 'Times New Roman');
        end
    end
    sgtitle([type_label ' PIT Histograms'], 'FontName', 'Times New Roman');
    print(f, fullfile(out_dir, [filename_prefix '.png']), '-dpng', '-r300');
    close(f);
end
