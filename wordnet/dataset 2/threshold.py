import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')

class PureGMMAnalyzer:
    def __init__(self, max_components=3, n_init=10, random_state=42):
        self.scaler = StandardScaler()
        self.max_components = max_components
        self.n_init = n_init
        self.random_state = random_state
        self.gmm = None
        self.X = None
        self.best_n_components = None
        self.model_metrics = {}
        
    def preprocess_data(self, data):
        """Basic preprocessing without any value filtering"""
        flattened = np.hstack(data)
        self.X = self.scaler.fit_transform(flattened.reshape(-1, 1))
        return self.X

    def optimize_gmm(self, data):
        """Model selection using only BIC"""
        X = self.preprocess_data(data)
        best_bic = np.inf
        best_gmm = None
        
        metrics_history = []
        
        for n_components in range(1, self.max_components + 1):
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=self.random_state,
                n_init=self.n_init
            )
            gmm.fit(X)
            
            bic = gmm.bic(X)
            metrics_history.append({
                'n_components': n_components,
                'bic': bic
            })
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                self.best_n_components = n_components
        
        self.gmm = best_gmm
        self.model_metrics = pd.DataFrame(metrics_history)
        return self.best_n_components

    def find_threshold(self):
        """Improved threshold detection using GMM components"""
        if not self.gmm or self.gmm.n_components < 2:
            return None

        # Sort components by mean
        means = self.gmm.means_.flatten()
        covars = self.gmm.covariances_.flatten()
        weights = self.gmm.weights_
        
        # Sort components by their means
        sorted_idx = np.argsort(means)
        means = means[sorted_idx]
        covars = covars[sorted_idx]
        weights = weights[sorted_idx]

        # Generate points for density estimation
        x = np.linspace(self.X.min(), self.X.max(), 1000)
        densities = np.zeros_like(x)

        # Calculate weighted density for each component
        for mu, var, w in zip(means, covars, weights):
            densities += w * stats.norm.pdf(x, mu, np.sqrt(var))

        # Find valleys between components
        valleys = []
        for i in range(len(means)-1):
            # Focus on region between consecutive means
            mask = (x >= means[i]) & (x <= means[i+1])
            if np.any(mask):
                valley_idx = mask.nonzero()[0][np.argmin(densities[mask])]
                valleys.append((x[valley_idx], densities[valley_idx]))

        if not valleys:
            # Fallback: use weighted average of component means
            threshold = np.average(means, weights=weights)
        else:
            # Use the most prominent valley (lowest density)
            threshold = min(valleys, key=lambda x: x[1])[0]

        # Convert back to original scale
        return float(self.scaler.inverse_transform([[threshold]])[0][0])

    def plot_analysis(self, original_data):
        """Visualization of the analysis"""
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Distribution with GMM Components
        ax1 = plt.subplot(131)
        self._plot_distribution(ax1)
        
        # 2. BIC Scores
        ax2 = plt.subplot(132)
        self._plot_bic(ax2)
        
        # 3. Threshold Analysis
        ax3 = plt.subplot(133)
        self._plot_threshold_analysis(ax3)
        
        plt.tight_layout()
        plt.show()
        
        # Interactive Plotly visualization
        self._create_interactive_plot()

    def _plot_distribution(self, ax):
        """Plot data distribution with GMM components"""
        x_range = np.linspace(self.X.min(), self.X.max(), 1000).reshape(-1, 1)
        
        sns.histplot(data=self.X, bins=30, stat='density', alpha=0.5, ax=ax)
        
        colors = sns.color_palette("husl", n_colors=self.gmm.n_components)
        for i, (mean, covar, weight) in enumerate(zip(
            self.gmm.means_, self.gmm.covariances_, self.gmm.weights_)):
            pdf = stats.norm.pdf(x_range, mean[0], np.sqrt(covar[0])) * weight
            ax.plot(x_range, pdf, color=colors[i], 
                   label=f'Component {i+1}')
        
        ax.set_title('Data Distribution with GMM Components')
        ax.legend()

    def _plot_bic(self, ax):
        """Plot BIC scores"""
        ax.plot(self.model_metrics['n_components'], 
               self.model_metrics['bic'], 
               marker='o', 
               label='BIC')
        
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('BIC Score')
        ax.set_title('Model Selection using BIC')
        ax.legend()

    def _plot_threshold_analysis(self, ax):
        """Plot threshold analysis"""
        threshold = self.find_threshold()
        scaled_threshold = self.scaler.transform([[threshold]])[0][0]
        
        sns.kdeplot(data=self.X, ax=ax)
        ax.axvline(scaled_threshold, color='r', linestyle='--', 
                  label=f'Threshold: {threshold:.3f}')
        
        ax.set_title('Threshold Analysis')
        ax.legend()

    def _create_interactive_plot(self):
        """Create interactive Plotly visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.X.flatten(),
            histnorm='probability density',
            name='Data',
            opacity=0.5
        ))
        
        x_range = np.linspace(self.X.min(), self.X.max(), 1000).reshape(-1, 1)
        for i, (mean, covar, weight) in enumerate(zip(
            self.gmm.means_, self.gmm.covariances_, self.gmm.weights_)):
            pdf = stats.norm.pdf(x_range, mean[0], np.sqrt(covar[0])) * weight
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=pdf.flatten(),
                name=f'Component {i+1}',
                mode='lines'
            ))
        
        threshold = self.find_threshold()
        scaled_threshold = self.scaler.transform([[threshold]])[0][0]
        fig.add_vline(x=scaled_threshold, 
                     line_dash="dash", 
                     line_color="red",
                     annotation_text=f"Threshold: {threshold:.3f}")
        
        fig.update_layout(
            title="Interactive GMM Analysis",
            xaxis_title="Value",
            yaxis_title="Density",
            hovermode='x'
        )
        
        fig.show()

def analyze_reviews(file_path):
    """Main analysis function"""
    data = pd.read_csv(file_path)
    data['entropy_values'] = data['entropy_values'].apply(ast.literal_eval)
    
    analyzer = PureGMMAnalyzer(max_components=3)
    n_components = analyzer.optimize_gmm(data['entropy_values'])
    threshold = analyzer.find_threshold()
    
    analyzer.plot_analysis(data['entropy_values'])
    
    return {
        'optimal_components': n_components,
        'threshold': threshold,
        'bic_scores': analyzer.model_metrics.to_dict('records')
    }

if __name__ == "__main__":
    results = analyze_reviews('rating_sa.csv')
    print("\nAnalysis Results:")
    print(f"Optimal number of components: {results['optimal_components']}")
    print(f"Computed threshold: {results['threshold']:.4f}")
    print("\nBIC Scores:")
    print(pd.DataFrame(results['bic_scores']))