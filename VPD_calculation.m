% =========================================================================
% Code Functionality Summary
%
% This MATLAB script performs the following tasks:
%
% 1. Reads Data: Loads 2-meter air temperature ('t2m') and 2-meter dewpoint 
%    temperature ('d2m') from two specific NetCDF ('.nc') files located on 
%    your local drive. It also reads coordinate data (time, latitude, longitude).
%
% 2. Data Processing:
%    * Converts the time variable from seconds since 1970 to a MATLAB 
%      'datetime' format.
%    * Converts temperature values from Kelvin to Celsius.
%
% 3. VPD Calculation:
%    * Iterates through each time step.
%    * Calculates the Saturation Vapor Pressure (using temperature) and the 
%      Actual Vapor Pressure (using dewpoint temperature) based on the 
%      Magnus formula.
%    * Computes the Vapor Pressure Deficit (VPD) by subtracting the actual 
%      vapor pressure from the saturation vapor pressure.
%
% 4. Saves Output: Creates a new NetCDF file ('vpd.nc'), defines 
%    the necessary dimensions (longitude, latitude, time) and attributes, 
%    and writes the calculated VPD data into it.
% =========================================================================

% Required Files:
% 1. H:\ERA5\VPD\2m temperature.nc
% 2. H:\ERA5\VPD\2m dewpoint temperature.nc

% File paths and filenames
tempFile = 'H:\ERA5\VPD\2m temperature.nc';
dewpFile = 'H:\ERA5\VPD\2m dewpoint temperature.nc';

% Read data from NC files
tempData = ncread(tempFile, 't2m'); % Temperature variable
dewpData = ncread(dewpFile, 'd2m'); % Dewpoint temperature variable
time = ncread(tempFile, 'valid_time'); % Time variable (units in seconds)
latitude = ncread(tempFile, 'latitude'); % Latitude
longitude = ncread(tempFile, 'longitude'); % Longitude

% Convert time to datetime type
timeOrigin = datetime(1970, 1, 1);
timeDates = timeOrigin + seconds(time);

% Initialize VPD matrix, dimensions are [longitude, latitude, time]
vpdData = zeros(size(tempData));

% Calculate VPD
for t = 1:length(timeDates)
    % Read temperature and dewpoint temperature for the current time step
    temp = tempData(:, :, t) - 273.15; % Convert to degrees Celsius
    dewp = dewpData(:, :, t) - 273.15; % Convert to degrees Celsius

    % Calculate saturation vapor pressure and actual vapor pressure (Unit: hPa)
    es_temp = 6.112 * exp(17.67 * temp ./ (temp + 243.5)); % Saturation vapor pressure from temperature
    es_dewp = 6.112 * exp(17.67 * dewp ./ (dewp + 243.5)); % Actual vapor pressure from dewpoint temperature

    % VPD Calculation
    vpd = es_temp - es_dewp;
    vpdData(:, :, t) = vpd; % Store the result
end

% Save VPD results to a new NC file
vpdFile = 'H:\ERA5\VPD\vpd.nc';

% Create file and define dimensions
nccreate(vpdFile, 'longitude', 'Dimensions', {'longitude', length(longitude)});
nccreate(vpdFile, 'latitude', 'Dimensions', {'latitude', length(latitude)});
nccreate(vpdFile, 'time', 'Dimensions', {'time', length(time)});
nccreate(vpdFile, 'VPD', 'Dimensions', {'longitude', length(longitude), 'latitude', length(latitude), 'time', length(time)});

% Write data
ncwrite(vpdFile, 'longitude', longitude);
ncwriteatt(vpdFile, 'longitude', 'units', 'degrees_east');
ncwriteatt(vpdFile, 'longitude', 'long_name', 'longitude');

ncwrite(vpdFile, 'latitude', latitude);
ncwriteatt(vpdFile, 'latitude', 'units', 'degrees_north');
ncwriteatt(vpdFile, 'latitude', 'long_name', 'latitude');

ncwrite(vpdFile, 'time', time);
ncwriteatt(vpdFile, 'time', 'units', 'seconds since 1970-01-01');
ncwriteatt(vpdFile, 'time', 'calendar', 'proleptic_gregorian');
ncwriteatt(vpdFile, 'time', 'long_name', 'time');

ncwrite(vpdFile, 'VPD', vpdData);
ncwriteatt(vpdFile, 'VPD', 'units', 'hPa');
ncwriteatt(vpdFile, 'VPD', 'long_name', 'Vapor Pressure Deficit');

disp('VPD calculation completed and saved to file');
