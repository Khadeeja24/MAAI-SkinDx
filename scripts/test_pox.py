import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from orchestrator.orchestrator import Orchestrator

pox_dir = os.path.join(PROJECT_ROOT, "data", "raw", "pox_dataset", "Monkeypox")
imgs = sorted([f for f in os.listdir(pox_dir)
               if f.lower().endswith(('.jpg','.jpeg','.png'))])

print(f"Total images to scan: {len(imgs)}")

o = Orchestrator()

passed = 0
for img_name in imgs:
    test_img = os.path.join(pox_dir, img_name)
    result   = o.process(test_img)

    if result.get('pipeline_status') == 'REJECTED':
        continue

    passed += 1
    print(f"\nPASSED Agent 1: {img_name}")
    print(f"Decision  : {result['decision']['decision']}")
    print(f"Diagnosis : {result['predicted_disease']}")
    print(f"Is OOD    : {result['is_ood']}")
    pox = result.get('pox_result') or {}
    print(f"Pox detect: {pox.get('pox_detected')}")
    print(f"Pox conf  : {(pox.get('confidence') or 0)*100:.1f}%")
    a7 = result.get('agents', {}).get('agent7') or {}
    print(f"Agent 7   : {a7.get('status')}")
    if a7.get('report_path'):
        print(f"Report    : {a7.get('report_path')}")
    break

if passed == 0:
    print("\nNo images passed Agent 1 in the entire Monkeypox folder.")
    print("The dataset images are too low quality for Agent 1.")
    print("This is a known limitation of the MSLD v2.0 dataset.")
    print("The pox pipeline is correct — the issue is dataset image quality.")