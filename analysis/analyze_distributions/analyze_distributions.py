import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from statsmodels.stats.weightstats import ztest
from pathlib import Path
import pandas as pd

class GrainDataset():
    def __init__(self, diameters_path=None, diameters=None, name:str = None, base_path:str ='', centroids = None):

        if diameters_path is not None:
            self.diameters = self.read_diameters(f'{base_path}/{diameters_path}')
            self.original_file_path = diameters_path

        elif diameters is not None:
            self.diameters = diameters
            self.original_file_path = name
        else:
           self.diameters = [1]

        self.name = name
        self.log_diameters = np.log(self.diameters)
        self.mean = np.mean(self.diameters)
        self.reduced_diameters = self.diameters/self.mean
        self.log_reduced_diameters = np.log(self.reduced_diameters)
        self.ln_fit = self.fit_lognormal()
        self.mean_area = self.mean_area()
        self.areas = [(d/2)**2 for d in self.diameters]
        self.log_areas = np.log(self.areas)
        self.reduced_areas = self.areas/self.mean_area
        self.log_reduced_areas = np.log(self.reduced_areas)
        self.diameter_of_mean_area = np.sqrt(self.mean_area/3.14159)*2
        self.centroids = centroids

    def read_diameters(self, diameters_path):
        with open(diameters_path, 'r') as file:
            return [float(line.strip()) for line in file]   
        
    def plot_distribution(self, bins=50, frequency=False, log=False, fit = False, xlim = False, ylim = False):
        # Generate the histogram
        if log:
            data = self.log_diameters
        else:
            data = self.diameters

        plt.hist(data, bins=bins, density=[not frequency], edgecolor='black')

        plt.title('Diameter Distribution for ' + self.original_file_path)
        if log:
            plt.xlabel('Log Diameter')
        else:
            plt.xlabel('Diameter')
        if frequency:
            plt.ylabel('Frequency')
        else:
            plt.ylabel('Probability Density')

        # Generate x values for plotting
        
        
        # Plot the fitted lognormal distribution
        if fit:
            if log:
                x = np.linspace(0, max(self.log_diameters), 50)
                plt.plot(x, self.fit_normal().pdf(x), 'r-', label='Normal Fit')
            else:
                x = np.linspace(0, max(self.diameters), 50)
                plt.plot(x, self.ln_fit.pdf(x), 'r-', label='Lognormal Fit')
                
        plt.legend()
        
        # Set custom axes limits if specified
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.show()


    def check_lognormality(self):
        # Perform Shapiro-Wilk test
        stat, pvalue = stats.shapiro(self.log_diameters) #I know a guy named Ben 
        print("Shapiro-Wilk Test:")
        print(f"Statistic: {stat}")
        print(f"P-value: {pvalue}")
        
        # Perform Anderson-Darling test
        result = stats.anderson(self.log_diameters)
        print("\nAnderson-Darling Test:")
        print(f"Statistic: {result.statistic}")
        print(f"Critical Values: {result.critical_values}")
        print(f"Significance Levels: {result.significance_level}\n")
        print(f"Normality: {result.statistic < result.critical_values[2]}")

    def fit_lognormal(self, reduced = False):
        # Fit lognormal distribution
        if reduced:
            shape, loc, scale = stats.lognorm.fit(self.reduced_diameters)
        else:
            shape, loc, scale = stats.lognorm.fit(self.diameters)

        lognormal_dist = stats.lognorm(shape, loc, scale)
        return lognormal_dist
    
    def fit_normal(self):
        # Fit normal distribution
        mean, std = stats.norm.fit(self.log_diameters)
        normal_dist = stats.norm(mean, std)
        return normal_dist

    def filter_smalls(self, threshold:float = 0.05):
        self.filtered_out = [d for d in self.diameters if d**2 < threshold*(self.mean**2)]
        self.diameters = [d for d in self.diameters if d**2 > threshold*(self.mean**2)]
        self.log_diameters = np.log(self.diameters)
        self.mean = np.mean(self.diameters)
        self.reduced_diameters = self.diameters/self.mean
        self.log_reduced_diameters = np.log(self.reduced_diameters)
        self.ln_fit = self.fit_lognormal()
        
    def unfilter(self):
        self.diameters = self.diameters + self.filtered_out
        self.log_diameters = np.log(self.diameters)
        self.mean = np.mean(self.diameters)
        self.reduced_diameters = self.diameters/self.mean
        self.log_reduced_diameters = np.log(self.reduced_diameters)
        self.ln_fit = self.fit_lognormal()

    def mean_area(self):
        return 3.14159 * np.mean([(d/2)**2 for d in self.diameters])

    def dump_diameters(self, save_path:str):
        with open(save_path, 'w') as file:
            for diameter in self.diameters:
                file.write(f"{diameter}\n")
        print(f"Diameters saved to {save_path}")

def get_distribution_from_folder(folder:str, base_path:str = None):
    folder = Path(folder)
    diameters = []
    centroids = []
    areas = []
    for file in folder.glob("*.csv"):
        if file.stem[0]=='.':
            continue
        df = pd.read_csv(file)
        areas += list(df['Area'].values)
        centroids += list(df[['Centroid X', 'Centroid Y']].values)
    
    diameters = [2*(area/np.pi)**0.5 for area in areas]
        
    return GrainDataset(diameters=diameters, name=folder.name, base_path=base_path, centroids=centroids)

    
def compare_distributions(dataset1, dataset2 , lognormal=False, area = False):
    
    print('#-------------------------------------------------------------------#')
    print(f"Comparing \n{dataset1.original_file_path}. mean = {np.mean(dataset1.diameters)}, N = {len(dataset1.diameters)} and \n{dataset2.original_file_path}. mean = {np.mean(dataset2.diameters)}, N = {len(dataset2.diameters)}")


    if lognormal:
        if area:
            ds_test_1 = dataset1.log_areas
            ds_test_2 = dataset2.log_areas
            ds_test_red_1 = dataset1.log_reduced_areas
            ds_test_red_2 = dataset2.log_reduced_areas
            
        else:
            ds_test_1 = dataset1.log_diameters
            ds_test_2 = dataset2.log_diameters
            ds_test_red_1 = dataset1.log_reduced_diameters
            ds_test_red_2 = dataset2.log_reduced_diameters

    else:
        if area:
            ds_test_1 = dataset1.areas
            ds_test_2 = dataset2.areas
            ds_test_red_1 = dataset1.reduced_areas
            ds_test_red_2 = dataset2.reduced_areas

        else:
            ds_test_1 = dataset1.diameters
            ds_test_2 = dataset2.diameters
            ds_test_red_1 = dataset1.reduced_diameters
            ds_test_red_2 = dataset2.reduced_diameters



    # Perform t-test
    t_stat, t_pvalue = stats.ttest_ind(ds_test_1, ds_test_2)
    print("\nT-Test:")
    print(f"T statistic: {t_stat}")
    print(f"T p-value: {t_pvalue}")
    ztest_ = ztest(ds_test_1, ds_test_2)
    print(f"Z-Test stat: {ztest_[0]}")
    print(f"Z-Test p-value: {ztest_[1]}")

    # Perform KS test

    ks_stat, ks_pvalue = stats.kstest(ds_test_1, ds_test_2)
    print("KS Test:")
    print(f"KS statistic: {ks_stat}")
    print(f"KS p-value: {ks_pvalue}")
    

    
    # Perform CVM test
    cvm = stats.cramervonmises_2samp(ds_test_1, ds_test_2)
    print("\nCVM-Test:")
    cvm_stat = cvm.statistic
    cvm_pvalue = cvm.pvalue
    print(f"CVM statistic : {cvm.statistic}")
    print(f"CVM p-value : {cvm.pvalue}")

    print('#----------------------- REDUCED ------------------------------#')


    # Perform t-test
    t_stat_red, t_pvalue_red = stats.ttest_ind(ds_test_red_1, ds_test_red_2)
    print("\nT-Test:")
    print(f"T statistic REDUCED D: {t_stat_red}")
    print(f"T p-value REDUCED D: {t_pvalue_red}")
    ztest_red = ztest(ds_test_red_1, ds_test_red_2) 
    print(f"Z-Test stat REDUCED D: {ztest_red[0]}")
    print(f"Z-Test p-value REDUCED D: {ztest_red[1]}")

    ks_stat_red, ks_pvalue_red = stats.kstest(ds_test_red_1, ds_test_red_2)
    print("\nKS Test:")
    print(f"KS statistic REDUCED : {ks_stat_red}")
    print(f"KS p-value: {ks_pvalue_red}")
    

    
    # Perform CVM test
    cvm_red = stats.cramervonmises_2samp(ds_test_red_1, ds_test_red_2)
    print("\nCVM-Test:")
    print(f"CVM statistic REDUCED : {cvm_red.statistic}")
    print(f"CVM p-value REDUCED : {cvm_red.pvalue}")
    print('#-------------------------------------------------------------------#')


    return {'t_stat':t_stat, 't_pval':t_pvalue, 
            'ks_stat':ks_stat, 'ks_pvalue': ks_pvalue, 
            'cvm_stat':cvm_stat, 'cvm_pvalue': cvm_pvalue,
            't_stat_red':t_stat_red, 't_pvalue_red':t_pvalue_red, 
            'ks_stat_red':ks_stat_red, 'ks_pvalue_red':ks_pvalue_red, 
            'cvm_stat_red':cvm_red.statistic, 'cvm_pvalue_red':cvm_red.pvalue}
    #pervform C
def plot_distros(datasets:list, hist = False, fit = False, reduced = False, log = False, xlim = (0,4), ylim=(0,1.2), nbins = 50, 
                 save_folder = '/Users/matthew/Documents/research/papers/my_papers/yolo-vs-unet/gbdetect_yolo_unet_2/figures',
                 display = False):
    colors_hist = ['black', 'r', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors_line = ['black', 'r', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    symbols = ['o', 'v', 's', 'v', 'd', 'p', 'h', '8', 'D']
    ii = 0
    plt.close()
    for dataset in datasets:
        data = dataset['data']
        plt_data = []
        if log:
            if reduced:
                plt_data = data.log_reduced_diameters
                save_name = f'{datasets[1]['name']}_vs_{datasets[0]['name']}_reduced_log_fit={bool(fit)}'
                plt.xlabel('Reduced Log Diameter')
            else:
                plt_data = data.log_diameters
                save_name = f'{datasets[1]['name']}_vs_{datasets[0]['name']}_log_fit={bool(fit)}'
                plt.xlabel('Log Diameter')
        else:
            if reduced:
                plt_data = data.reduced_diameters
                save_name = f'{datasets[1]['name']}_vs_{datasets[0]['name']}_reduced_fit={bool(fit)}'
                plt.xlabel('Reduced Diameter')
            else:
                plt_data = data.diameters
                save_name = f'{datasets[1]['name']}_vs_{datasets[0]['name']}_fit={bool(fit)}'
                plt.xlabel('Diameter [nm]')
        
            

        if hist:
            bins = int(max(plt_data)/xlim[1]*nbins)
            histogram, bins = np.histogram(plt_data, bins=bins, density=True)

            # Calculate the center of each bin
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Plot using scatter
            plt.scatter(bin_centers, histogram, marker=symbols[ii], c=colors_hist[ii], label=f'{dataset['name']} Histogram')
            #plt.hist(plt_data, 'o',bins=50, density=True, edgecolor='black', label=dataset['name'], color=colors_hist[ii], alpha=1.0)    
        

        if fit:  
            x = np.linspace(0, max(plt_data), 250)
            if log:
                fit = data.fit_normal(reduced = reduced)      
            else:
                fit = data.fit_lognormal(reduced = reduced)
            plt.plot(x, fit.pdf(x), '-', c = colors_line[ii])
            x = np.linspace(0, max(plt_data), 250)

        ii+=1
    
    plt.gca().tick_params(axis='both', direction = 'in', top = True, right = True)
    plt.ylabel('Probability Density')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    if save_folder:
        plt.savefig(f'{save_folder}/{save_name}.png', dpi = 500)
    print(f'Saved {save_name}.png')

    if display:
        plt.show()
    else:
        print('Display set to False. Not showing plot')
        plt.close()

    plt.close()

if __name__ == "__main__":
    # Define the paths to the diameter file
    dataset_60min  =  GrainDataset(diameters_path="/Volumes/Samsung_T5/Matthew/TEM/Al-324/225C_60min_diameters_lhg-retrace-100epoch.txt")
    dataset_15min  =  GrainDataset(diameters_path="/Volumes/Samsung_T5/Matthew/TEM/Al-324/225C_15min_diameters_lhg-retrace-100epoch.txt")
    dataset_120min = GrainDataset(diameters_path="/Volumes/Samsung_T5/Matthew/TEM/Al-324/225C_120min_diameters_lhg-retrace-100epoch.txt")
    dataset_asdep  =  GrainDataset(diameters_path="/Volumes/Samsung_T5/Matthew/TEM/Al-324/asdep_diameters_lgh-retrace-100epoch.txt")
    
    compare_distributions(dataset_60min, dataset_15min, lognormal=True)
    compare_distributions(dataset_60min, dataset_120min, lognormal=True)
    compare_distributions(dataset_60min, dataset_asdep, lognormal=True)

    compare_distributions(dataset_15min, dataset_120min, lognormal=True)
    compare_distributions(dataset_15min, dataset_asdep, lognormal=True)

    compare_distributions(dataset_120min, dataset_asdep, lognormal=True)

