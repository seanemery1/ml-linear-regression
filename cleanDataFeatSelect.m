function [daySelect, dayCategory] = cleanDataFeatSelect(day)
% This function tidies up the data (unnormalizes normalized values) in
% addition to selecting features (while generating graphs to support
% rationale behind feature selections)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% A Brief Digression on Time-series Data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Residual Plot Over Time (days/instant) %%%
% The significance of this will be discussed in the technical report
% Generating graph/plots
hold on
regCntDays = fitlm(day,'cnt~instant');
scatter(regCntDays.Variables.instant, regCntDays.Residuals.Standardized);
plot(regCntDays.Variables.instant, regCntDays.Residuals.Standardized);
title('Residuals against Time (Days, 2011-2012)');
xlabel('Time (Days, 2011-2012)');
ylabel('Residuals (Standardized)');
str = ['R^2 = ' num2str(regCntDays.Rsquared.Adjusted)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 14, 'verticalalignment', 'top',...
    'horizontalalignment', 'left');
hold off
figure
%%%%%%

%%% Growth (Total Riders) Over Time (days/instant) %%%
% The significance of this will be discussed in the technical report
% Generating graph/plots
hold on
plot(regCntDays);
title('Growth of Total Riders over Time (Days, 2011-2012)');
xlabel('Time (Days, 2011-2012)');
ylabel('Total Riders');
%xl = xline(2328,'-.','Average','DisplayName','Average Sales');
xline(365, '-.', {'2011', '2012'},  'DisplayName', '2011/2012 Marker',...
    'LabelHorizontalAlignment', 'center');
str = ['R^2 = ' num2str(regCntDays.Rsquared.Adjusted)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 14, 'verticalalignment', 'top',...
    'horizontalalignment', 'left');
hold off
figure

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dependent Variable and Features Selection %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determining best Y between casual, registered, and cnt %%%
% From the results below, count was chosen (explanation in technical
% report)
subplot(1,3,1);
[b1, b0, res] = regressXY(day.casual, day.registered, ' Casual Riders ',...
    ' Registered Riders ');
subplot(1,3,2);
[b1, b0, res] = regressXY(day.registered, day.cnt, ' Registered Riders ',...
    ' Total Riders ');
subplot(1,3,3);
[b1, b0, res] = regressXY(day.cnt, day.casual,...
    ' Total Riders ', ' Casual Riders ');
figure
%%%%%%


%%% The Issues With season and weathersit in their Current Form %%%
% Explanation of issue in technical report
subplot(1,2,1)
[b1, b0, res] = regressXY(day.season, day.cnt,...
    ' season (winter > spring > summer > fall) ', ' Total Riders ');
subplot(1,2,2)
[b1, b0, res] = regressXY(day.weathersit, day.cnt,...
    ' Weather (clear > cloudy > light) ', ' Total Riders ');
figure

%%% Creating Seasonal Dummy Variables for Regression %%%
% Initializing size of seasons
% summer is excluded to avoid perfect multicollinearity/dummy variable trap
winter = day.holiday;
spring = day.holiday;
% summer = day.holiday;
fall = day.holiday;

% Zeroing the seasons' column values
for iRow = 1:height(day)
    winter(iRow, 1) = 0;
    spring(iRow, 1) = 0;
    % summer(iRow, 1) = 0;
    fall(iRow, 1) = 0;
end

% Turning 0s to 1s for the right seasonal dummy variables
for iRow = 1:height(day)
    if day{iRow, 3} == 1 % if season = winter
        winter(iRow, 1) = 1;
    elseif day{iRow, 3} == 2 % if season = spring
        spring(iRow, 1) = 1;
%     elseif day{iRow, 3} == 3 % if season = summer
%         summer(iRow, 1) = 1;
    elseif day{iRow, 3} == 4 % if season = fall
        fall(iRow, 1) = 1;
    end
end

% Concatenate day, winter, spring, summer, fall to table daySeaWeath
day = [day(:,:), array2table(winter),...
    array2table(spring), array2table(fall)]; % array2table(summer)


%%% Creating Weather Dummy Variables for Regression %%%
% Initializing size of weather conditions
% clear is excluded to avoid perfect multicollinearity/dummy variable trap
% clear = day.holiday;
cloudy = day.holiday;
light = day.holiday;
% heavy = day.holiday; does not exist

% Zeroing the weathersits' column values
for iRow = 1:height(day)
    % clear(iRow, 1) = 0
    cloudy(iRow, 1) = 0;
    light(iRow, 1) = 0;
end

% Turning 0s to 1s for the right weathersit dummy variables
for iRow = 1:height(day)
%     if day{iRow, 9} == 1 % if weathersit = clear
%         winter(iRow, 1) = 1;
    if day{iRow, 9} == 2 % if weathersit = cloudy
        cloudy(iRow, 1) = 1;
    elseif day{iRow, 9} == 3 % if weathersit = light
        light(iRow,1) = 1;
    end
end

% Concatenate  daySeaWeath, cloud, light, to table daySeaWeath
day = [day, array2table(cloudy),...
    array2table(light)]; % array2table()

 
%%% Selecting Between temp and atemp %%%
% temp : Normalized temperature in Celsius. The values are derived via
% (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39
% atemp: Normalized feeling temperature in Celsius. The values are derived
% via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50
% From results below, atemp (Feeling Temp) is selected.
subplot(1,2,1)
[b1, b0, res] = regressXY(day.temp, day.atemp,...
    ' Temperature (normalized) ', ' Feeling Temperature (normalized) ');
% Unnormalized temp values will be used instead as the interpretation
% of unnormalized temp is more meaningful than normalized temp (it is
% easier to generalize DC data to LA).
subplot(1,2,2)
[b1, b0, res] = regressXY((day.temp*(39--8))+-8, day.atemp*66-16,...
    ' Temperature ', ' Feeling Temperature ');
figure

day.atemp = (day.atemp*(50--16))+-16;

%%%%%%


%%% Remaing features selection via coefficient comparisons %%%
% hum: Normalized humidity. The values are divided to 100 (max)
% windspeed: Normalized wind speed. The values are divided to 67 (max)
subplot(1,2,1)
[b1, b0, res] = regressXY(day.hum, day.windspeed, ' Humidity ',...
    ' Windspeed ');
% Unnormalized windspeed/hum  values will be used instead as the
% interpretation of unnormalized temp is more meaningful than normalized
% windspeed/hum (it is easier to generalize DC data to LA).
% *This will not be graphed*
day.hum = day.hum*100;
day.windspeed = day.windspeed*100;

% Colinearity check for workingday/holiday
subplot(1,2,2)
[b1, b0, res] = regressXY(day.workingday, day.holiday,...
    ' Working Day (Yes/No)', ' Holiday (Yes/No) ');


%%% Final feature selection %%%
% instant, dteday, season, yr,  mnth, holiday, weekday, workingday,
% weathersit, temp, atemp, hum, windspeed, casual, registered, cnt, winter
% spring, fall, cloudy, light
% \/\/\/\/\/\/\/\/\/\/\/\/\/\/
% atemp, hum, windspeed, workingday, holiday, winter, spring, fall,
% cloudy, light, cnt
% nondummy
daySelect = day(:,[1, 11:13, 8, 6, 17:21, 16]);

day.workingday = categorical(day.workingday, [0, 1], {'not work day', 'work day'});
day.holiday = categorical(day.holiday, [0, 1], {'not holiday', 'holiday'});
day.season = categorical(day.season, [1, 2, 3, 4], {'winter', 'spring',...
    'summer', 'fall'});
day.weathersit = categorical(day.weathersit, [1, 2, 3], {'clear', 'cloudy', 'light'});
%dayDummy = day(:,10:12);
dayCategory = day(:,[1,11:13, 8, 6, 3, 9, 16]);
end