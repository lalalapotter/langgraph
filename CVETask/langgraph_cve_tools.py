# langgraph_cve_tools.py

import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- Global Configuration ---
WORKSPACE = os.path.join(os.getcwd(), 'workspace/doc')
print(WORKSPACE)
REPORTS_DIR = os.path.join(WORKSPACE, 'reports')
print(REPORTS_DIR)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKSPACE, 'debug'), exist_ok=True)

# --- Tool Definitions ---

@tool
def trivy_scanner(image_name: str) -> str:
    """
    Scans a Docker image using Trivy and saves the output to a JSON file.
    Args:
        image_name: The full name of the Docker image to scan.
    Returns:
        A JSON string with the path to the output file or an error.
    """
    sanitized_name = image_name.replace(':', '_').replace('/', '_')
    output_path = os.path.join(WORKSPACE, f'{sanitized_name}_cves.json')
    command = ['trivy', 'image', '--format', 'json', '--output', output_path, image_name]
    logging.info(f"Running Trivy scan for {image_name}...")
    logging.info(command)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return json.dumps({"output_path": output_path})
    except FileNotFoundError:
        return json.dumps({"error": "Trivy command not found. Please ensure Trivy is installed."})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"Trivy scan failed for '{image_name}'. Stderr: {e.stderr}"})

@tool
def json_to_csv_converter(json_file_path: str) -> str:
    """
    Converts a Trivy JSON report file to a CSV file.
    Args:
        json_file_path: The path to the Trivy JSON report.
    Returns:
        A JSON string with the path to the output CSV file or an error.
    """
    if not os.path.exists(json_file_path):
        return json.dumps({"error": f"Input file not found at {json_file_path}"})
    
    csv_file_path = json_file_path.replace('.json', '.csv')
    headers = ["VulnerabilityID", "PkgName", "InstalledVersion", "FixedVersion", "Severity", "Title", "Description", "PrimaryURL"]
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f_json, \
             open(csv_file_path, 'w', encoding='utf-8', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=headers)
            writer.writeheader()
            data = json.load(f_json)
            if 'Results' not in data or data['Results'] is None:
                return json.dumps({"output_path": csv_file_path, "status": "empty"})
            for result in data.get('Results', []):
                for vuln in result.get('Vulnerabilities', []):
                    writer.writerow({h: str(vuln.get(h, "")).replace('\n', ' ') for h in headers})
        return json.dumps({"output_path": csv_file_path})
    except Exception as e:
        return json.dumps({"error": f"Error during JSON to CSV conversion: {e}"})

class CveClassifierInput(BaseModel):
    target_csv_path: str = Field(description="The file path for the target image's CVE CSV report.")
    base_csv_path: str = Field(description="The file path for the base image's CVE CSV report.")

@tool(args_schema=CveClassifierInput)
def cve_classifier(target_csv_path: str, base_csv_path: str) -> dict:
    """
    Reads two CVE CSV reports (target and base) and classifies the vulnerabilities.
    """
    def read_cves_from_csv(file_path: str) -> Dict[str, dict]:
        cves = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if cve_id := row.get("VulnerabilityID"):
                        cves[cve_id] = row
        except FileNotFoundError:
            return {}
        return cves

    target_cves = read_cves_from_csv(target_csv_path)
    base_cves = read_cves_from_csv(base_csv_path)
    base_cve_ids = set(base_cves.keys())
    
    type1_cves, type2_cves, type3_cves = [], [], []
    for cve_id, cve_data in target_cves.items():
        if not cve_data.get("FixedVersion"):
            type1_cves.append(cve_data)
        elif cve_id in base_cve_ids:
            type2_cves.append(cve_data)
        else:
            type3_cves.append(cve_data)
            
    return {"type1_cves": type1_cves, "type2_cves": type2_cves, "type3_cves_to_analyze": type3_cves}

@tool
def cve_report_generator(classified_cves: dict) -> str:
    """
    Generates .trivyignore files and a summary report from classified CVE lists.
    Args:
        classified_cves: A dictionary containing lists of CVEs, keyed by 'type1_cves', 'type2_cves', and 'type3_results'.
    """
    def _write_ignore_file(file_path: Path, ignore_list: List[Dict], header: str):
        if not ignore_list: return 0
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {header}\n# Auto-generated by LangGraph CVE Analyzer\n")
            f.write("vulnerabilities:\n")
            for entry in ignore_list:
                if not entry.get('id'): continue
                f.write(f"  - id: {entry['id']}\n")
                if entry.get('url'): f.write(f"    url: {entry['url']}\n")
                f.write(f"    statement: {entry['reason']}\n")
        return len(ignore_list)
    def _write_combined_ignore_file(file_path: Path, ignore_list: List[Dict], header: str):
                if not ignore_list: 
                    return 0
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # 检查文件是否存在，决定是否需要写入header
                write_header = not file_path.exists()
                with open(file_path, 'a', encoding='utf-8') as f:
                    if write_header:
                        f.write(f"# {header}\n# Auto-generated by Qwen-Agent\n")
                        f.write("vulnerabilities:\n")
                        
                    for entry in ignore_list:
                        f.write(f"- id: {entry['id']}\n")
                        if entry.get('url'): 
                            f.write(f"  url: {entry['url']}\n")
                        f.write(f"  statement: {entry['reason']}\n")
                return len(ignore_list)
    final_messages = []
    all_ignores = []

    # Process Type 1
    type1_ignores = [{'id': c.get('VulnerabilityID', ''), 'url':c.get('PrimaryURL',''), 'reason': "No fix available. Monitoring for updates."} for c in classified_cves.get('type1_cves', [])]
    count1 = _write_ignore_file(Path(WORKSPACE, 'trivyignore-type1.yaml'), type1_ignores, "Type 1: No Fix Available")
    if count1 > 0:
        final_messages.append(f"Generated {count1} Type-1 rules -> trivyignore-type1.yaml")
        all_ignores.extend(type1_ignores)

    # Process Type 2
    type2_ignores = [{'id': c.get('VulnerabilityID', ''), 'url':c.get('PrimaryURL',''), 'reason': "Inherited from base image. Monitoring for base image updates."} for c in classified_cves.get('type2_cves', [])]
    count2 = _write_ignore_file(Path(WORKSPACE, 'trivyignore-type2.yaml'), type2_ignores, "Type 2: Inherited from Base Image")
    if count2 > 0:
        final_messages.append(f"Generated {count2} Type-2 rules -> trivyignore-type2.yaml")
        all_ignores.extend(type2_ignores)

    # Process Type 3
    type3_results = classified_cves.get('type3_results', [])
    logging.info(f"Processing {len(type3_results)} expert analysis results for report generation.")
    
    type3_ignores = [{'id': i.get('cve', {}).get('VulnerabilityID'), 'url': i.get('cve', {}).get('PrimaryURL'), 'reason': f"Analyzed: Not relevant. {i.get('analysis', {}).get('analysis', 'N/A')}"} for i in type3_results if i.get('analysis', {}).get('whether_relevant', 'Yes').lower() != 'yes']
    count3 = _write_ignore_file(Path(WORKSPACE, 'trivyignore-type3.yaml'), type3_ignores, "Type 3: Analyzed as Not Relevant")
    if count3 > 0:
        final_messages.append(f"Generated {count3} Type-3 ignore rules -> trivyignore-type3.yaml")
        all_ignores.extend(type3_ignores)
 
    type3_relevant = [i for i in type3_results if i.get('analysis', {}).get('whether_relevant', 'No').lower() == 'yes']
    count3_relevant = _write_ignore_file(Path(WORKSPACE, 'relevant.yaml'), type3_relevant, "Analyzed by agent.")
    suggestions = [f"[{i.get('cve', {}).get('VulnerabilityID')}/{i.get('cve', {}).get('PkgName')}]: {i.get('analysis', {}).get('suggestion')}" for i in type3_relevant if i.get('analysis', {}).get('suggestion')]
    
    report_path = Path(REPORTS_DIR, 'actionable_vulnerabilities_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Actionable Vulnerabilities Requiring Fixes (Type-3 Analysis)\n" + "="*70 + "\n\n")
        if suggestions:
            f.write("Recommended Actions:\n\n")
            f.writelines(f"{s}\n\n" for s in suggestions)
        else:
            f.write("No actionable suggestions were generated from the analysis.\n")
    final_messages.append(f"Generated report for {len(suggestions)} actionable CVEs -> {report_path.name}")
    
    # Write combined ignore file
    if all_ignores:
        _write_ignore_file(Path(WORKSPACE, '.trivyignore'), all_ignores, "Combined Ignore Rules")
        final_messages.append(f"Generated combined .trivyignore with {len(all_ignores)} rules.")

    if not final_messages:
        return "Report generation complete. No new rules or reports were created."
        
    return "Report generation complete.\n- " + "\n- ".join(final_messages)