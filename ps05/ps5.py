# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import matplotlib.pyplot as plt
import re


# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models = []
    for deg in degs:
        coeffs = pylab.polyfit(x, y, deg)
        models.append(coeffs)
    return models


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    # Calculate the residual sum of squares
    rss = ((y - estimated) ** 2).sum()
    # Calculate the total sum of squares
    tss = ((y - y.mean()) ** 2).sum()
    return 1 - rss / tss

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        p = pylab.poly1d(model)
        estimated = p(x)
        r_sq = r_squared(y, estimated)

        plt.xticks(x)
        plt.scatter(x, y, c='b')
        plt.plot(x, estimated, c='r')
        plt.title('')
        plt.xlabel('Year')
        plt.ylabel('Temperature (degrees Celsius)')

        if len(model) == 2:
            se = se_over_slope(x, y, estimated, model)
            plt.title(f'Polynomial of degree {len(model) - 1}, R^2: {r_sq},\nSE over slope: {se}')
        else:
            plt.title(f'Polynomial of degree {len(model) - 1}, R^2: {r_sq}')
        plt.show()

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    result = []
    for year in years:
        inner_result = []
        for city in multi_cities:
            inner_result.append(climate.get_yearly_temp(city=city, year=year).mean())
        result.append(pylab.array(inner_result).mean())
    return pylab.array(result)

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    means = []
    for i, val in enumerate(y):
        idx = i - window_length + 1
        mean = pylab.append(y[idx if idx >= 0 else 0:i], val).mean()
        means.append(mean)
    return pylab.array(means)

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    squared_error = ((y - estimated) ** 2).sum()
    return pylab.sqrt(squared_error / len(y))

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    result = []
    for year in years:
        inner_result = []
        for city in multi_cities:
            temp = climate.get_yearly_temp(city, year)
            inner_result.append(temp)
        mean = pylab.array(inner_result).mean(axis=0)
        std = pylab.std(mean)
        std = pylab.std(mean)
        result.append(std)
    return pylab.array(result)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        p = pylab.poly1d(model)
        estimated = p(x)
        rmse_val = rmse(y, estimated)

        plt.xticks(x)
        plt.scatter(x, y, c='b')
        plt.plot(x, estimated, c='r')
        plt.title('')
        plt.xlabel('Year')
        plt.ylabel('Temperature (degrees Celsius)')
        plt.title(f'Polynomial of degree {len(model) - 1}, RMSE: {rmse_val}')
        plt.show()

if __name__ == '__main__':
    climate = Climate(filename='./data.csv')
    years = list(TRAINING_INTERVAL)
    x = pylab.array(years)

    # Part A.4
    temps = []
    for i in years:
        temp = climate.get_daily_temp(
            city='NEW YORK',
            year=i,
            month=1,
            day=10
        )
        temps.append(temp)

    y = pylab.array(temps)
    models = generate_models(x, y, [1])
    evaluate_models_on_training(x, y, models)

    temps = []
    for i in years:
        temp = climate.get_yearly_temp(
            city='NEW YORK',
            year=i,
        )
        temps.append(temp)

    y = pylab.array([t.mean() for t in temps])
    models = generate_models(x, y, [1])
    evaluate_models_on_training(x, y, models)

    # What difference does choosing a specific day to plot the data for versus calculating the yearly average
    # have on our graphs (i.e., in terms of the R^2 values and the fit of the resulting curves)?
    # Interpret the results.
    # A: Choosing the yearly average yields an R^2 value of 0.18 versus 0.05 when choosing a specific day.
    # This indicates that the yearly average has less variance and can be better captured by the degree-one
    # polynomial fit. However, the data exhibit a lot of variance, hence the low R^2 values.
    # The ratio of SE is 0.6 and 0.3 respectively, which again indicates that choosing yearly
    # averages is a much better strategy since it's below the 0.5 threshold to be considered statistically significant.

    # Why do you think these graphs are so noisy? Which one is more noisy?
    # A: The first graph (choosing a specific day) is more noisy than the second one (yearly averages).
    # The reason for this is that a single day of each year may not be a very good indicator of
    # the overall year due to random noise. Averaging the values gets rid of some of the noise.

    # How do these graphs support or contradict the claim that global warming is leading to an increase
    # in temperature? The slope and the standard error-to-slope ratio could be helpful in thinking about this.
    # A: The first graph doesn't support the claim as the standard error ratio is > 0.5, meaning that
    # the trend in the data cannot be considered significant.
    # The second graph supports the claim that the temperature is increasing as the line fits the data better,
    # explains more of the variance and the standard error ratio is < 0.5 which indicates a statistically
    # significant result.

    # Part B
    y = gen_cities_avg(climate, CITIES, years)
    models = generate_models(x, y, [1])
    evaluate_models_on_training(x, y, models)

    # How does this graph compare to the graphs from part A (i.e., in terms of the R^2 values, the fit of the resulting
    # curves, and whether the graph supports/contradicts our claim about global warming)? Interpret the results.
    # A: The values are a lot less noisy as evidenced by an R^2 value of 0.74. The model explains a
    # much larger proportion of the variance of the data. The SE ratio is 0.08 indicating a statistically
    # significant result. The graph strongly supports the claim about global warming.

    # Why do you think this is the case?
    # A: Generally, more data means more accurate model. Since we've incorporated data from 21 different cities
    # across the US, we've included a range of climates as opposed to simply using a single city.

    # How would we expect the results to differ if we used 3 different cities?
    # A: The results would probably be better than using a single city but worse than using all 21.

    # What about 100 different cities?
    # A: The model would become even more reliable, however there is a risk of over-fitting if the cities
    # happen to be clustered near one another. Using a test set to verify the model generalizes
    # well to data it hasn't seen yet would alleviate this.

    # How would the results have changed if all 21 cities were in the same region
    # of the United States (for ex., New England)?
    # A: The model would be very accurate on the training data, however it would not generalize well
    # to other cities since all of the cities would tend to have a similar climate.

    # Part C
    city_averages = gen_cities_avg(climate, CITIES, years)
    y = moving_average(city_averages, 5)
    models = generate_models(x, y, [1])
    evaluate_models_on_training(x, y, models)

    # How does this graph compare to the graphs from part A and B (i.e., in terms of the R^2 values,
    # the fit of the resulting curves, and whether the graph supports/contradicts our claim about global warming)?
    # Interpret the results.
    # A: The R^2 value is 0.92 indicating that the model fits the data very well and that there isn't much variance
    # in the data. The graph strongly supports our claim. The model is even more reliable than before.

    # Why do you think this is the case?
    # A: Using a moving average of 5 years means we're averaging over the past 5 years every time. This
    # decreases the effect of outliers, making it easier to fit a line to the data.

    # Part D.2
    city_averages = gen_cities_avg(climate, CITIES, years)
    y = moving_average(city_averages, 5)
    models = generate_models(x, y, [1, 2, 20])
    evaluate_models_on_training(x, y, models)

    # How do these models compare to each other?
    # A: They all fit the data similarly well. The higher the degree of the polynomial, the better the fit,
    # however I suspect this is due to over-fitting in case of the degree-20 polyomial.

    # Which one has the best R^2? Why?
    # A: The degree-20 polynomial model has the best R^2 value. Due to the high degree of the polynomial,
    # the model is very flexible and can fit the data extremely well.

    # Which model best fits the data? Why?
    # A: The degree-20 polynomial model fits the data best for the reasons mentioned above.

    years_test = list(TESTING_INTERVAL)
    x_test = pylab.array(years_test)
    city_averages_test = gen_cities_avg(climate, CITIES, years_test)
    y_test = moving_average(city_averages_test, 5)
    evaluate_models_on_testing(x_test, y_test, models)

    # How did the different models perform? How did their RMSEs compare?
    # A: The RMSE is lowest for the degree-one polynomial, larger for degree 2 and very high for degree 20.
    # The first model performs quite well on the test data with an RMSE of 0.08, the second one has poorer performance
    # with an RMSE of 0.21 and the last model performs very poorly with an RMSE of 1.49.

    # Which model performed the best? Which model performed the worst? Are they the same as those in part D.2.I? Why?
    # A: The simplest model performed the best whereas the most complex one performed the worst. This is due to the
    # fact that the complex model can very closely fit the training data, which leads to over-fitting, making it unable
    # to generalize to data it hasn't seen yet.

    # If we had generated the models using the A.4.II data (i.e. average annual temperature of New York City)
    # instead of the 5-year moving average over 22 cities, how would the prediction results 2010-2015 have changed?
    # A: The model from A.4.II was a lot less accurate making it very unlikely that it would perform well
    # on unseen data.

    # Part E
    std_devs = gen_std_devs(climate, CITIES, years)
    y = moving_average(std_devs, 5)
    models = generate_models(x, y, [1])
    evaluate_models_on_training(x, y, models)

    # Does the result match our claim (i.e., temperature variation is getting larger over these years)?
    # A: No, the result goes against the claim. The graph shows that the temperature variation is getting
    # smaller over time

    # Can you think of ways to improve our analysis?
    # A: We could fit a higher degree polynomial although we would run the risk of overfitting.
