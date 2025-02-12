import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_json_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                all_data.append({
                    'filename': filename,
                    'data': data
                })
    return all_data

def extract_metrics(json_data):
    metrics = []
    for entry in json_data:
        metric = {
            'query': entry['query'],
            'bleu_score': entry['bleu_score'],
            'trust_score': entry['trust_score'],
            'rouge_f1': entry['rouge_scores'][0]['rouge-1']['f'],
            'passed': entry['passed']
        }
        metrics.append(metric)
    return metrics
def analyze_specific_question(all_runs_data, question_index, output_dir):

    question_dir = os.path.join(output_dir, f'question_{question_index+1}')
    os.makedirs(question_dir, exist_ok=True)
    
    # Datenextraktion
    question_data = []
    for run in all_runs_data:
        if question_index < len(run['data']):
            entry = run['data'][question_index]
            metrics = {
                'run': run['filename'],
                'bleu_score': entry['bleu_score'],
                'trust_score': entry['trust_score'],
                'rouge_f1': entry['rouge_scores'][0]['rouge-1']['f'],
                'passed': entry['passed']
            }
            question_data.append(metrics)
    
    df = pd.DataFrame(question_data)
    
    # filename sortieren
    df = df.sort_values(by='run').reset_index(drop=True)

    # frage vom 1. durchlauf
    question_text = all_runs_data[0]['data'][question_index]['query']
    
    # 1. Score distribution plot
    plt.figure(figsize=(10, 6))
    scores_df = df[['bleu_score', 'rouge_f1']].melt()
    sns.boxplot(x='variable', y='value', data=scores_df)
    plt.title(f'Score Verteilung der Frage {question_index+1}\n"{question_text}"')
    plt.savefig(os.path.join(question_dir, 'score_distribution.png'))
    plt.close()
    
    # 2. Pass/Fail ratio
    plt.figure(figsize=(8, 6))
    pass_rate = df['passed'].mean()
    plt.pie([pass_rate, 1-pass_rate], labels=['Pass', 'Fail'], 
            autopct='%1.1f%%', colors=['green', 'red'])
    plt.title(f'Pass/Fail Ratio für Frage {question_index+1}')
    plt.savefig(os.path.join(question_dir, 'pass_fail_ratio.png'))
    plt.close()

    # 3. Liniendiagramm: Metriken über verschiedene Durchläufe hinweg
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['bleu_score'], marker='o', linestyle='-', label='BLEU Score')
    plt.plot(df.index, df['rouge_f1'], marker='s', linestyle='-', label='ROUGE-1 F1')
    plt.plot(df.index, df['trust_score'], marker='d', linestyle='-', label='Trust Score')

    plt.xlabel('Durchlauf')
    plt.ylabel('Score')
    plt.title(f'Metriken über verschiedene Durchläufe hinweg für Frage {question_index+1}\n"{question_text}"')
    plt.legend()
    plt.grid(True)
    plt.xticks(df.index, df['run'], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(question_dir, 'metric_trends.png'))
    plt.close()

    # standardverteilung
    plt.figure(figsize=(10, 6))

    # Falls zu wenige Datenpunkte da sind, stattdessen Histogramm nutzen
    if len(df) < 5:
        sns.histplot(df['bleu_score'], bins=10, kde=True, label='BLEU Score', alpha=0.6)
        sns.histplot(df['rouge_f1'], bins=10, kde=True, label='ROUGE-1 F1', alpha=0.6)
        sns.histplot(df['trust_score'], bins=10, kde=True, label='Trust Score', alpha=0.6)
    else:
        sns.kdeplot(df['bleu_score'], label='BLEU Score', fill=True, bw_adjust=0.5)
        sns.kdeplot(df['rouge_f1'], label='ROUGE-1 F1', fill=True, bw_adjust=0.5)
        sns.kdeplot(df['trust_score'], label='Trust Score', fill=True, bw_adjust=0.5)

    plt.xlabel('Score')
    plt.ylabel('Dichte')
    plt.title(f'Standardverteilung der Scores für Frage {question_index+1}\n"{question_text}"')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(question_dir, 'score_distribution_kde.png'))
    plt.close()


    summary = {
        'question': question_text,
        'metrics': {
            'avg_bleu': df['bleu_score'].mean(),
            'avg_rouge': df['rouge_f1'].mean(),
            'avg_trust': df['trust_score'].mean(),
            'pass_rate': pass_rate,
            'number_of_runs': len(df)
        }
    }
    

    with open(os.path.join(question_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def create_visualizations(all_runs_data, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    for run in all_runs_data:
        metrics = extract_metrics(run['data'])
        run_df = pd.DataFrame(metrics)
        run_df['run'] = run['filename']
        all_metrics.append(run_df)
    
    combined_df = pd.concat(all_metrics)
    
    # 1. Box plot Score
    plt.figure(figsize=(12, 6))
    scores_df = combined_df[['bleu_score', 'rouge_f1', 'trust_score']].melt()
    sns.boxplot(x='variable', y='value', data=scores_df)
    plt.title('Distribution of Scores Across All Runs')
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()
    
    # 2. Heatmap 
    plt.figure(figsize=(15, 8))
    avg_scores = combined_df.groupby('query')[['bleu_score', 'rouge_f1', 'trust_score']].mean()
    sns.heatmap(avg_scores, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Average Scores per Question')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_scores_heatmap.png'))
    plt.close()
    
    # 3. Pass/Fail ratio per run
    plt.figure(figsize=(10, 6))
    pass_rates = combined_df.groupby('run')['passed'].mean()
    pass_rates.plot(kind='bar')
    plt.title('Pass Rate per Run')
    plt.xlabel('Run')
    plt.ylabel('Pass Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_rates.png'))
    plt.close()
    

    summary = {
        'overall_stats': {
            'avg_bleu': combined_df['bleu_score'].mean(),
            'avg_rouge': combined_df['rouge_f1'].mean(),
            'avg_trust': combined_df['trust_score'].mean(),
            'overall_pass_rate': combined_df['passed'].mean()
        },
        'per_run_stats': combined_df.groupby('run').agg({
            'bleu_score': 'mean',
            'rouge_f1': 'mean',
            'trust_score': 'mean',
            'passed': 'mean'
        }).to_dict()
    }
    
    # json speichern
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def main():

    input_directory = 'C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\'  
    output_directory = 'C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\evaluation_results\\' 
    

    print("Loading JSON files...")
    all_runs_data = load_json_files(input_directory)
    
    while True:
        print("\nWähle eine Option:")
        print("1. Analyse aller Fragen")
        print("2. Analyse einer spezifischen Frage")
        print("3. Beenden")
        
        choice = input("Deine Wahl (1-3): ")
        
        if choice == "1":
            print("\nErstelle Visualisierungen für alle Fragen...")
            summary = create_visualizations(all_runs_data, output_directory)
            print("\nGesamtstatistik:")
            print(f"Durchschnittlicher BLEU Score: {summary['overall_stats']['avg_bleu']:.4f}")
            print(f"Durchschnittlicher ROUGE-1 F1: {summary['overall_stats']['avg_rouge']:.4f}")
            print(f"Durchschnittlicher Trust Score: {summary['overall_stats']['avg_trust']:.4f}")
            print(f"Gesamte Bestehensquote: {summary['overall_stats']['overall_pass_rate']*100:.2f}%")
            
        elif choice == "2":
           
            num_questions = len(all_runs_data[0]['data'])
            print(f"\nVerfügbare Fragen (0-{num_questions-1}):")
            for i in range(num_questions):
                print(f"{i}: {all_runs_data[0]['data'][i]['query']}")
            
            question_index = int(input("\nWähle eine Frage (Nummer): "))
            if 0 <= question_index < num_questions:
                summary = analyze_specific_question(all_runs_data, question_index, output_directory)
                print(f"\nStatistik für Frage {question_index}:")
                print(f"Frage: {summary['question']}")
                print(f"Durchschnittlicher BLEU Score: {summary['metrics']['avg_bleu']:.4f}")
                print(f"Durchschnittlicher ROUGE-1 F1: {summary['metrics']['avg_rouge']:.4f}")
                print(f"Durchschnittlicher Trust Score: {summary['metrics']['avg_trust']:.4f}")
                print(f"Bestehensquote: {summary['metrics']['pass_rate']*100:.2f}%")
            else:
                print("Ungültige Fragennummer!")
                
        elif choice == "3":
            break
        
        else:
            print("Ungültige Eingabe!")

if __name__ == "__main__":
    main()