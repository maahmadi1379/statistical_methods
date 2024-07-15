import numpy as np
from scipy import stats

"""
A module containing classes for performing various statistical tests.

Classes:
    TTests: Contains static methods for performing t-tests.
    FTests: Contains static methods for performing F-tests.
"""


class TTests:
    """
    Class for performing t-tests.

    Static methods:
        one_sample_t_test: Performs a one-sample t-test.
        paired_sample_t_test: Performs a paired sample t-test.
        independent_two_sample_t_test: Performs an independent two-sample t-test.
    """

    @staticmethod
    def one_sample_t_test(sample, population_mean):
        """
        Perform a one-sample t-test.

        Parameters:
            sample (np.ndarray): The sample data.
            population_mean (float): The hypothesized mean of the population from which the sample was drawn.

        Returns:
            tuple: A tuple containing the t-statistic and the p-value.
        """
        t_statistic, p_value = stats.ttest_1samp(sample, population_mean)
        return t_statistic, p_value

    @staticmethod
    def paired_sample_t_test(sample1, sample2):
        """
        Perform a paired sample t-test.

        Parameters:
            sample1 (np.ndarray): The first set of sample data.
            sample2 (np.ndarray): The second set of sample data, which should correspond to the same subjects measured twice.

        Returns:
            tuple: A tuple containing the t-statistic and the p-value.
        """
        t_statistic, p_value = stats.ttest_rel(sample1, sample2)
        return t_statistic, p_value

    @staticmethod
    def independent_two_sample_t_test(sample1, sample2):
        """
        Perform an independent two-sample t-test.

        Parameters:
            sample1 (np.ndarray): The first set of sample data.
            sample2 (np.ndarray): The second set of sample data.

        Returns:
            tuple: A tuple containing the t-statistic and the p-value.
        """
        t_statistic, p_value = stats.ttest_ind(sample1, sample2)
        return t_statistic, p_value


class FTests:
    """
    Class for performing F-tests.

    Static methods:
        one_way_anova_f_test: Performs a one-way ANOVA F-test.
        two_sample_variance_f_test: Performs a two-sample variance ratio F-test.
    """

    @staticmethod
    def one_way_anova_f_test(*groups):
        """
        Perform a one-way ANOVA F-test.

        Parameters:
            *groups (list of np.ndarray): Groups of data to compare.

        Returns:
            tuple: A tuple containing the F-statistic and the p-value.
        """
        f_statistic, p_value = stats.f_oneway(*groups)
        return f_statistic, p_value

    @staticmethod
    def two_sample_variance_f_test(sample1, sample2):
        """
        Perform a two-sample variance ratio F-test.

        Parameters:
            sample1 (np.ndarray): The first set of sample data.
            sample2 (np.ndarray): The second set of sample data.

        Returns:
            tuple: A tuple containing the F-statistic and the p-value.
        """
        f_statistic = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
        dfn, dfd = len(sample1) - 1, len(sample2) - 1
        p_value = 1 - stats.f.cdf(f_statistic, dfn, dfd)
        return f_statistic, p_value


if __name__ == '__main__':
    # Examples:

    # Set the seed
    np.random.seed(123)

    # 1. One-Sample T-Test
    sample = np.random.normal(loc=5, scale=2, size=100)
    population_mean = 4.5
    t_stat, p_val = TTests.one_sample_t_test(sample, population_mean)
    print("One-Sample T-Test:")
    print(f"T-statistic: {round(t_stat, 5)}, p-value: {round(p_val, 5)}\n")

    # 2. Paired Sample T-Test
    before_treatment = np.random.normal(loc=10, scale=2, size=50)
    after_treatment = before_treatment + np.random.normal(loc=0.5, scale=1, size=50)
    t_stat, p_val = TTests.paired_sample_t_test(before_treatment, after_treatment)
    print("Paired Sample T-Test:")
    print(f"T-statistic: {round(t_stat, 5)}, p-value: {round(p_val, 5)}\n")

    # 3. Independent Two-Sample T-Test
    group1 = np.random.normal(loc=5, scale=2, size=100)
    group2 = np.random.normal(loc=6, scale=2, size=100)
    t_stat, p_val = TTests.independent_two_sample_t_test(group1, group2)
    print("Independent Two-Sample T-Test:")
    print(f"T-statistic: {round(t_stat, 5)}, p-value: {round(p_val, 5)}\n")

    # 4. One-Way ANOVA F-Test
    group1 = np.random.normal(loc=5, scale=2, size=100)
    group2 = np.random.normal(loc=6, scale=2, size=100)
    group3 = np.random.normal(loc=5.5, scale=2, size=100)
    f_stat, p_val = FTests.one_way_anova_f_test(group1, group2, group3)
    print("One-Way ANOVA F-Test:")
    print(f"F-statistic: {round(f_stat, 5)}, p-value: {round(p_val, 5)}\n")

    # 5. Two-Sample for Variances F-Test
    sample1 = np.random.normal(loc=5, scale=2, size=100)
    sample2 = np.random.normal(loc=5, scale=3, size=100)
    f_stat, p_val = FTests.two_sample_variance_f_test(sample1, sample2)
    print("Two-Sample for Variances F-Test:")
    print(f"F-statistic: {round(f_stat, 5)}, p-value: {round(p_val, 5)}\n")
