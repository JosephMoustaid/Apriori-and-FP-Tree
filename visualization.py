import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import networkx as nx
from matplotlib.patches import FancyBboxPatch as FBP


class AlgorithmVisualizer:
    
    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 10)
        
    def visualize_itemset_distribution(self, frequent_itemsets, algorithm_name="Algorithm"):
        itemset_sizes = {}
        for itemset, support in frequent_itemsets.items():
            size = len(itemset)
            if size not in itemset_sizes:
                itemset_sizes[size] = []
            itemset_sizes[size].append(support)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        sizes = sorted(itemset_sizes.keys())
        counts = [len(itemset_sizes[s]) for s in sizes]
        
        bars = ax1.bar(sizes, counts, color=self.colors[0], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Itemset Size (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Frequent Itemsets', fontsize=12, fontweight='bold')
        ax1.set_title(f'{algorithm_name}: Itemset Size Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(sizes)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for size in sizes:
            supports = itemset_sizes[size]
            positions = np.random.normal(size, 0.04, size=len(supports))
            ax2.scatter(positions, supports, alpha=0.6, s=50, color=self.colors[size % len(self.colors)])
        
        ax2.set_xlabel('Itemset Size (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Support Count', fontsize=12, fontweight='bold')
        ax2.set_title(f'{algorithm_name}: Support by Itemset Size', fontsize=14, fontweight='bold')
        ax2.set_xticks(sizes)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{algorithm_name.lower()}_itemset_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_top_itemsets(self, frequent_itemsets, top_n=15, algorithm_name="Algorithm"):
        sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        labels = [' & '.join(str(item) for item in itemset) for itemset, _ in sorted_itemsets]
        supports = [support for _, support in sorted_itemsets]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
        bars = ax.barh(range(len(labels)), supports, color=colors_gradient, edgecolor='black', linewidth=1.2)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Support Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{algorithm_name}: Top {top_n} Frequent Itemsets', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, support) in enumerate(zip(bars, supports)):
            width = bar.get_width()
            ax.text(width + max(supports)*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{int(support)}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{algorithm_name.lower()}_top_itemsets.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_association_rules(self, rules_data, top_n=20, algorithm_name="Algorithm"):
        if not rules_data:
            print("No rules to visualize")
            return
            
        df = pd.DataFrame(rules_data).sort_values(by=['lift', 'confidence'], ascending=False).head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rule_labels = [f"{' & '.join(str(i) for i in row['antecedent'])} → {' & '.join(str(i) for i in row['consequent'])}" 
                      for _, row in df.iterrows()]
        
        colors_conf = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df)))
        axes[0, 0].barh(range(len(df)), df['confidence'], color=colors_conf, edgecolor='black', linewidth=1)
        axes[0, 0].set_yticks(range(len(df)))
        axes[0, 0].set_yticklabels(rule_labels, fontsize=8)
        axes[0, 0].set_xlabel('Confidence', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Association Rules by Confidence', fontsize=12, fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        colors_lift = plt.cm.cool(np.linspace(0.2, 0.9, len(df)))
        axes[0, 1].barh(range(len(df)), df['lift'], color=colors_lift, edgecolor='black', linewidth=1)
        axes[0, 1].set_yticks(range(len(df)))
        axes[0, 1].set_yticklabels(rule_labels, fontsize=8)
        axes[0, 1].set_xlabel('Lift', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Association Rules by Lift', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        scatter = axes[1, 0].scatter(df['support'], df['confidence'], 
                                     s=df['lift']*50, c=df['lift'], 
                                     cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1)
        axes[1, 0].set_xlabel('Support', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Support vs Confidence (bubble size = Lift)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('Lift', fontsize=10, fontweight='bold')
        
        metrics_data = [
            ['Metric', 'Min', 'Max', 'Mean', 'Median'],
            ['Support', f"{df['support'].min():.4f}", f"{df['support'].max():.4f}", 
             f"{df['support'].mean():.4f}", f"{df['support'].median():.4f}"],
            ['Confidence', f"{df['confidence'].min():.4f}", f"{df['confidence'].max():.4f}", 
             f"{df['confidence'].mean():.4f}", f"{df['confidence'].median():.4f}"],
            ['Lift', f"{df['lift'].min():.4f}", f"{df['lift'].max():.4f}", 
             f"{df['lift'].mean():.4f}", f"{df['lift'].median():.4f}"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=metrics_data, cellLoc='center', loc='center',
                                colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(metrics_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(metrics_data)):
            table[(i, 0)].set_facecolor('#E8F5E9')
            table[(i, 0)].set_text_props(weight='bold')
        
        axes[1, 1].set_title('Rule Metrics Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(f'{algorithm_name}: Association Rules Analysis (Top {top_n})', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{algorithm_name.lower()}_association_rules.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_rules_network(self, rules_data, top_n=15, algorithm_name="Algorithm"):
        if not rules_data:
            print("No rules to visualize")
            return
        
        df = pd.DataFrame(rules_data).sort_values(by=['lift'], ascending=False).head(top_n)
        
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            antecedent_str = ' & '.join(str(i) for i in row['antecedent'])
            consequent_str = ' & '.join(str(i) for i in row['consequent'])
            
            G.add_edge(antecedent_str, consequent_str, 
                      weight=row['confidence'], 
                      lift=row['lift'],
                      support=row['support'])
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        lifts = [G[u][v]['lift'] for u, v in edges]
        
        node_colors = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if out_degree > in_degree:
                node_colors.append('#FF6B6B')
            elif in_degree > out_degree:
                node_colors.append('#4ECDC4')
            else:
                node_colors.append('#95E1D3')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.9, 
                              edgecolors='black', linewidths=2, ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        edge_colors = plt.cm.plasma(np.array(lifts) / max(lifts))
        
        for (u, v), color, width in zip(edges, edge_colors, weights):
            ax.annotate('',
                       xy=pos[v], xycoords='data',
                       xytext=pos[u], textcoords='data',
                       arrowprops=dict(arrowstyle='->', color=color,
                                     lw=width*3, alpha=0.7,
                                     connectionstyle="arc3,rad=0.1"))
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                   norm=plt.Normalize(vmin=min(lifts), vmax=max(lifts)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Lift', fontsize=12, fontweight='bold')
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=10, label='Antecedent (mostly)', markeredgecolor='black', markeredgewidth=1),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                      markersize=10, label='Consequent (mostly)', markeredgecolor='black', markeredgewidth=1),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95E1D3', 
                      markersize=10, label='Both', markeredgecolor='black', markeredgewidth=1)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title(f'{algorithm_name}: Association Rules Network (Top {top_n})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{algorithm_name.lower()}_rules_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_algorithms(self, apriori_itemsets, fpgrowth_itemsets, 
                          apriori_time, fpgrowth_time):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        apriori_sizes = {}
        for itemset in apriori_itemsets.keys():
            size = len(itemset)
            apriori_sizes[size] = apriori_sizes.get(size, 0) + 1
            
        fpgrowth_sizes = {}
        for itemset in fpgrowth_itemsets.keys():
            size = len(itemset)
            fpgrowth_sizes[size] = fpgrowth_sizes.get(size, 0) + 1
        
        all_sizes = sorted(set(list(apriori_sizes.keys()) + list(fpgrowth_sizes.keys())))
        apriori_counts = [apriori_sizes.get(s, 0) for s in all_sizes]
        fpgrowth_counts = [fpgrowth_sizes.get(s, 0) for s in all_sizes]
        
        x = np.arange(len(all_sizes))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, apriori_counts, width, label='Apriori', 
                      color='#FF6B6B', alpha=0.8, edgecolor='black')
        axes[0, 0].bar(x + width/2, fpgrowth_counts, width, label='FP-Growth', 
                      color='#4ECDC4', alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Itemset Size', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Itemsets by Size', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(all_sizes)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        total_itemsets = [len(apriori_itemsets), len(fpgrowth_itemsets)]
        algorithms = ['Apriori', 'FP-Growth']
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = axes[0, 1].bar(algorithms, total_itemsets, color=colors, 
                             alpha=0.8, edgecolor='black', linewidth=2)
        axes[0, 1].set_ylabel('Total Itemsets', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Total Frequent Itemsets', fontsize=12, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, total_itemsets):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(count)}', ha='center', va='bottom', 
                          fontsize=12, fontweight='bold')
        
        execution_times = [apriori_time, fpgrowth_time]
        bars = axes[1, 0].bar(algorithms, execution_times, color=colors, 
                             alpha=0.8, edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Execution Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for bar, time_val in zip(bars, execution_times):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{time_val:.4f}s', ha='center', va='bottom', 
                          fontsize=11, fontweight='bold')
        
        comparison_data = [
            ['Metric', 'Apriori', 'FP-Growth'],
            ['Total Itemsets', str(len(apriori_itemsets)), str(len(fpgrowth_itemsets))],
            ['Execution Time', f'{apriori_time:.4f}s', f'{fpgrowth_time:.4f}s'],
            ['Max Itemset Size', str(max(apriori_sizes.keys())), str(max(fpgrowth_sizes.keys()))],
            ['Avg Itemset Size', f"{sum(k*v for k,v in apriori_sizes.items())/sum(apriori_sizes.values()):.2f}",
             f"{sum(k*v for k,v in fpgrowth_sizes.items())/sum(fpgrowth_sizes.values()):.2f}"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=comparison_data, cellLoc='center', 
                                loc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(len(comparison_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        table[(0, 1)].set_facecolor('#FF6B6B')
        table[(0, 2)].set_facecolor('#4ECDC4')
        
        for i in range(1, len(comparison_data)):
            table[(i, 0)].set_facecolor('#E8F5E9')
            table[(i, 0)].set_text_props(weight='bold')
        
        axes[1, 1].set_title('Algorithm Comparison', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Apriori vs FP-Growth Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_fp_tree(self, fp_tree_root, header_table, max_depth=3, max_children=10):
        node_count = {'total': 0, 'rendered': 0}
        
        def count_nodes(node, depth=0):
            if depth > max_depth:
                return
            node_count['total'] += len(node.children)
            for child in node.children.values():
                count_nodes(child, depth + 1)
        
        count_nodes(fp_tree_root)
        
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.axis('off')
        
        positions = {}
        nodes_to_draw = []
        
        def calculate_positions(node, depth=0, x_pos=0, x_width=1.0, parent_pos=None):
            if depth > max_depth:
                return
            
            children_items = sorted(node.children.items(), 
                                   key=lambda x: x[1].count, reverse=True)
            
            num_children = min(len(children_items), max_children)
            if num_children == 0:
                return
            
            children_items = children_items[:num_children]
            
            y_pos = 1.0 - (depth * 0.25)
            x_step = x_width / (num_children + 1)
            
            for i, (item, child) in enumerate(children_items):
                child_x = x_pos + (i + 1) * x_step
                child_pos = (child_x, y_pos)
                positions[id(child)] = child_pos
                
                nodes_to_draw.append({
                    'pos': child_pos,
                    'item': item,
                    'count': child.count,
                    'parent_pos': parent_pos,
                    'depth': depth
                })
                
                node_count['rendered'] += 1
                
                calculate_positions(child, depth + 1, child_x - x_step/2, 
                                  x_step, child_pos)
        
        root_pos = (0.5, 1.05)
        positions[id(fp_tree_root)] = root_pos
        
        calculate_positions(fp_tree_root, 0, 0, 1.0, root_pos)
        
        for node_info in nodes_to_draw:
            if node_info['parent_pos']:
                ax.plot([node_info['parent_pos'][0], node_info['pos'][0]], 
                       [node_info['parent_pos'][1], node_info['pos'][1]], 
                       'gray', linewidth=1.5, alpha=0.4, zorder=1)
        
        for node_info in nodes_to_draw:
            color = plt.cm.tab20(hash(node_info['item']) % 20)
            circle = plt.Circle(node_info['pos'], 0.015, color=color, 
                               ec='black', linewidth=1, zorder=3, alpha=0.8)
            ax.add_patch(circle)
            
            if node_info['depth'] < 2:
                label = f"{node_info['item']}\n({node_info['count']})"
                ax.text(node_info['pos'][0], node_info['pos'][1], label, 
                       ha='center', va='center', fontsize=7, 
                       fontweight='bold', zorder=4)
        
        root_circle = plt.Circle(root_pos, 0.02, color='#FFD700', 
                                ec='black', linewidth=2, zorder=3)
        ax.add_patch(root_circle)
        ax.text(root_pos[0], root_pos[1], 'ROOT', ha='center', va='center', 
               fontsize=9, fontweight='bold', zorder=4)
        
        sorted_items = sorted(header_table.items(), 
                            key=lambda x: x[1]['count'], reverse=True)[:15]
        
        legend_x = 0.02
        legend_y_start = 0.95
        ax.text(legend_x, legend_y_start, 'Top Items:', fontsize=10, 
               fontweight='bold', ha='left', transform=ax.transAxes)
        
        for i, (item, data) in enumerate(sorted_items):
            y = legend_y_start - (i + 1) * 0.05
            color = plt.cm.tab20(hash(item) % 20)
            rect = Rectangle((legend_x, y - 0.015), 0.015, 0.03, 
                           facecolor=color, edgecolor='black', linewidth=0.8,
                           transform=ax.transAxes, alpha=0.8)
            ax.add_patch(rect)
            ax.text(legend_x + 0.025, y, f"{item}: {data['count']}", 
                   fontsize=8, va='center', ha='left', transform=ax.transAxes)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.15)
        
        title = f'FP-Tree Structure (Depth ≤ {max_depth}, Top {max_children} branches per node)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        stats_text = f'Showing {node_count["rendered"]} of {node_count["total"]} total nodes'
        ax.text(0.5, -0.02, stats_text, ha='center', fontsize=9, 
               style='italic', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig('fp_tree_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ FP-Tree visualization saved ({node_count['rendered']} nodes shown)")


def visualize_apriori_results(apriori_obj):
    visualizer = AlgorithmVisualizer()
    
    if not apriori_obj.all_frequent_itemsets:
        print("No frequent itemsets to visualize")
        return
    
    flat_itemsets = {}
    for k, itemsets in apriori_obj.all_frequent_itemsets.items():
        flat_itemsets.update(itemsets)
    
    visualizer.visualize_itemset_distribution(flat_itemsets, "Apriori")
    visualizer.visualize_top_itemsets(flat_itemsets, top_n=15, algorithm_name="Apriori")


def visualize_fpgrowth_results(fpgrowth_obj, frequent_itemsets):
    visualizer = AlgorithmVisualizer()
    
    if not frequent_itemsets:
        print("No frequent itemsets to visualize")
        return
    
    visualizer.visualize_itemset_distribution(frequent_itemsets, "FP-Growth")
    visualizer.visualize_top_itemsets(frequent_itemsets, top_n=15, algorithm_name="FP-Growth")
    
    if hasattr(fpgrowth_obj, 'tree_root') and hasattr(fpgrowth_obj, 'header_table'):
        visualizer.visualize_fp_tree(fpgrowth_obj.tree_root, fpgrowth_obj.header_table)


def visualize_rules(rules_list, algorithm_name="Algorithm"):
    visualizer = AlgorithmVisualizer()
    visualizer.visualize_association_rules(rules_list, top_n=20, algorithm_name=algorithm_name)
    visualizer.visualize_rules_network(rules_list, top_n=15, algorithm_name=algorithm_name)
