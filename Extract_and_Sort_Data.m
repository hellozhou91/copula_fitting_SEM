% Complete Code

% ==============================================================================
% FILE REQUIREMENTS:
% 1. NetCDF Data File: 
%    - A .nc file containing 'longitude', 'latitude', 'time', and 'VPD' variables.
%    - Path defined in variable 'ncFilePath'.
%
% 2. Shapefiles (7 Regions):
%    - A set of .shp files representing different climate zones (A to G).
%    - Paths defined in 'shp_files_regions' cell array.
%
% SCRIPT FUNCTIONALITY:
% 1. Configuration: Sets time range (2000-2023) and file paths.
% 2. Data Loading: Reads VPD, coordinate, and time data from the NetCDF file.
% 3. Pre-processing: Slices data to the specified time range and adjusts dimensions.
% 4. Spatial Masking: 
%    - Iterates through 7 regional Shapefiles.
%    - Creates a spatial mask to keep data only within these regions.
%    - Maps each grid point to a specific domain name (A-G).
% 5. NetCDF Export: Saves the clipped data (regions outside masks set to NaN) 
%    to a new NetCDF file.
% 6. Text Export: Extracts valid (non-NaN) data points, sorts them by 
%    Domain -> Time -> Latitude -> Longitude, and saves to a .txt file.
% 7. Visualization: Plots the original vs. clipped data for the first time step.
% ==============================================================================

% Read NetCDF file

% --- Configuration Start ---

% Modify the time range you want to process here
start_year = 2000;
end_year = 2023; % Example: You can adjust the end year to your needs

% The start year of the data file (usually does not need modification)
data_source_start_year = 2000;

% Regional Boundary Shapefiles
shp_files_regions = { % Define a cell array containing file paths for multiple region boundary shapefiles
    'H:\A.shp';
    'H:\B.shp';
    'H:\C.shp';
    'H:\D.shp';
    'H:\E.shp';
    'H:\F.shp';
    'H:\G.shp'
};

domain_names = {'A', 'B','C', 'D', 'E', 'F', 'G'}; % Define region names corresponding to the shapefiles

ncFilePath = 'Data.nc';

% --- Modification: Change output file path to relative path ---
% This saves generated files in the current MATLAB working directory to avoid permission issues
newNcFilePath = sprintf('VPD_%d-%d_0.5deg_clipped_7_region.nc', start_year, end_year);
outputTxtFilePath = sprintf('VPD_%d-%d_0.5deg_clipped_7_region_sorted.txt', start_year, end_year);

% --- Configuration End ---

% Load NetCDF data
disp('Loading NetCDF data...');

longitude = ncread(ncFilePath, 'longitude');
latitude = ncread(ncFilePath, 'latitude');
time_full = ncread(ncFilePath, 'time');
VPD_full = ncread(ncFilePath, 'VPD'); % Load complete VPD data

disp('NetCDF data loaded.');

% Extract data based on the set time range
disp('Extracting data based on time range...');

numMonths = 12;
start_index = (start_year - data_source_start_year) * numMonths + 1;
end_index = (end_year - data_source_start_year + 1) * numMonths;

if start_index < 1 || end_index > length(time_full) || start_index > end_index
    error('Specified time range (%d - %d) exceeds NetCDF file data range or is invalid.', start_year, end_year);
end

time = time_full(start_index:end_index);
VPD = VPD_full(:, :, start_index:end_index);

disp('Data extraction completed.');

% Adjust VPD dimensions to [Latitude, Longitude, Time]
VPD = permute(VPD, [2, 1, 3]); % Adjust dimension order of VPD

% Create latitude and longitude meshgrid
[LON, LAT] = meshgrid(longitude, latitude);

% --- Process multiple Shapefile regions ---
disp('Processing multiple Shapefile regions...');

% Initialize a string array of the same size as the grid to store the region name for each grid point
region_map = strings(size(LON));

% Initialize a logical mask to mark the union of all regions
combined_inRegion = false(size(LON));

% Loop through each region's shapefile
for r = 1:length(shp_files_regions)
    shpFilePath = shp_files_regions{r};
    domainName = domain_names{r};
    fprintf('Processing region: %s (%s)\n', domainName, shpFilePath);
    
    % Read Shapefile
    S = shaperead(shpFilePath);
    
    % Calculate mask for the current region
    current_region_mask = false(size(LON));
    for s = 1:length(S)
        % Use inpolygon to determine if grid points are inside the polygon
        current_region_mask = current_region_mask | inpolygon(LON, LAT, S(s).X, S(s).Y);
    end
    
    % Store the current region name in region_map
    % Assuming regions do not overlap; if they do, the later processed region covers the earlier one
    region_map(current_region_mask) = domainName;
    
    % Update the total region mask (union of all regions)
    combined_inRegion = combined_inRegion | current_region_mask;
end

disp('All Shapefile regions processed.');
disp(['Total data points inside regions: ', num2str(sum(combined_inRegion(:)))]);
disp(['Total data points outside regions: ', num2str(sum(~combined_inRegion(:)))]);

% --- Data Clipping and Saving ---

% Apply region mask (vectorized operation, replacing original loop)
% Create a 3D mask, marking all points outside regions as true
mask_3d = repmat(~combined_inRegion, [1, 1, size(VPD, 3)]);

VPD_clipped = VPD;

% Set data in masked areas (outside regions) to NaN
VPD_clipped(mask_3d) = NaN;

disp('Data clipping completed.');

% --- Create and Write New NetCDF File ---
disp('Preparing to create and write new NetCDF file...');

% --- Modification: Add file existence check and deletion logic ---
% Before creating the file, check if it exists. If so, delete it to avoid potential write conflicts.
if exist(newNcFilePath, 'file')
    try
        delete(newNcFilePath);
        disp(['Successfully deleted old file: ', newNcFilePath]);
    catch ME
        error('Unable to delete existing old file: %s. Please check file permissions or if the file is occupied by another program. Error details: %s', newNcFilePath, ME.message);
    end
end

disp(['Creating new NetCDF file: ', newNcFilePath]);
ncid = netcdf.create(newNcFilePath, 'NC_WRITE');

% Define dimensions
dimLongitude = netcdf.defDim(ncid, 'longitude', length(longitude));
dimLatitude = netcdf.defDim(ncid, 'latitude', length(latitude));
dimTime = netcdf.defDim(ncid, 'time', length(time)); % Use clipped time length

% Define variables
longitudeVar = netcdf.defVar(ncid, 'longitude', 'double', dimLongitude);
latitudeVar = netcdf.defVar(ncid, 'latitude', 'double', dimLatitude);
timeVar = netcdf.defVar(ncid, 'time', 'double', dimTime);
VPDVar = netcdf.defVar(ncid, 'VPD', 'double', [dimLongitude dimLatitude dimTime]);

% End definition mode
netcdf.endDef(ncid);

% Write NetCDF data (bulk write for efficiency)
netcdf.putVar(ncid, longitudeVar, longitude);
netcdf.putVar(ncid, latitudeVar, latitude);
netcdf.putVar(ncid, timeVar, time);

% VPD_clipped is [Latitude, Longitude, Time], while NetCDF variable VPD is defined as [Longitude, Latitude, Time]
% Therefore, dimensions need to be converted before writing
VPD_clipped_permuted = permute(VPD_clipped, [2, 1, 3]);
netcdf.putVar(ncid, VPDVar, VPD_clipped_permuted);

% Close NetCDF file
netcdf.close(ncid);
disp(['NetCDF file extracted and saved successfully! File path: ', newNcFilePath]);

% ----------------------------------------
% Collect, sort, and save non-NaN data to a text file
% ----------------------------------------
disp('Collecting valid data points within all regions...');

% Find linear indices of all non-NaN elements in the VPD_clipped matrix
valid_indices = find(~isnan(VPD_clipped));

% If no valid data points found, prompt and skip file writing
if isempty(valid_indices)
    disp('No valid data points found in the specified regions and time range. Text file will not be created.');
else
    % Use ind2sub to convert linear indices to subscripts for latitude, longitude, and time
    [lat_idx, lon_idx, time_idx] = ind2sub(size(VPD_clipped), valid_indices);
    
    % Extract region name for each valid data point from region_map
    % First, convert 2D latitude and longitude subscripts to linear indices for region_map
    region_map_linear_idx = sub2ind(size(region_map), lat_idx, lon_idx);
    point_domains = region_map(region_map_linear_idx);
    
    % Create a table to organize all data for sorting
    % Ensure all vectors are column vectors to correctly create table
    data_table = table(...
        longitude(lon_idx(:)), ... % Longitude
        latitude(lat_idx(:)), ...  % Latitude
        time(time_idx(:)), ...     % Time
        VPD_clipped(valid_indices), ... % VPD Value
        point_domains, ...         % Region Name
        'VariableNames', {'Longitude', 'Latitude', 'Time', 'VPD', 'domain'}...
    );
    
    disp('Data collection completed, sorting now...');
    
    % Sort in ascending order with priority "domain > Time > Latitude > Longitude"
    sorted_table = sortrows(data_table, {'domain', 'Time', 'Latitude', 'Longitude'});
    
    disp('Data sorting completed.');
    disp(['Writing sorted data to text file: ' outputTxtFilePath]);
    
    % Use writetable function to write sorted table to text file
    % writetable automatically handles file opening, header and data writing, and file closing
    try
        writetable(sorted_table, outputTxtFilePath, 'Delimiter', '\t', 'WriteVariableNames', true);
        disp(['Clipped and sorted non-NaN data successfully saved to: ' outputTxtFilePath]);
    catch ME
        error('Unable to write to file %s. Please check write permissions. Error message: %s', outputTxtFilePath, ME.message);
    end
end

% ----------------------------------------
% Visualize non-NaN data
% ----------------------------------------
disp('Generating data visualization charts...');

figure;
set(gcf, 'Name', 'Data Clipping Result Comparison', 'NumberTitle', 'off');

% Plot original data (first time step)
subplot(1, 2, 1);
pcolor(LON, LAT, VPD(:, :, 1)); % Use LON and LAT to ensure matching
shading interp;
colorbar;
title('Original Data (1st Time Step)');
xlabel('Longitude');
ylabel('Latitude');
axis equal tight; % Maintain aspect ratio and display compactly

% Plot clipped data (first time step)
subplot(1, 2, 2);
pcolor(LON, LAT, VPD_clipped(:, :, 1)); % Use LON and LAT to ensure matching
shading interp;
colorbar;
title('Clipped Data (All Regions Combined, 1st Time Step)');
xlabel('Longitude');
ylabel('Latitude');
axis equal tight; % Maintain aspect ratio and display compactly

disp('Data visualization completed.');
disp('All tasks completed!');
