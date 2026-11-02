import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.orchestrator import Orchestrator

orch = Orchestrator()

result = orch.process(
    image_path=r"C:\Users\khadeeja\Desktop\Projects\Skin\data\raw\ddi31\000001.png",
    symptoms={}
)

print("\n=== FIX 3 VERIFICATION ===")
print(f"Pipeline status      : {result['pipeline_status']}")
print(f"Pox override applied : {result.get('pox_override', 'N/A — pipeline failed early')}")
print(f"Final diagnosis      : {result.get('predicted_disease', 'N/A')}")

if result['pox_override']:
    pox_name = result['pox_result']['predicted_name']
    a4_name  = result['agents']['agent4'].get('predicted_name', 'N/A')
    print(f"\nPox disease          : {pox_name}")
    print(f"Agent 4 explained    : {a4_name}")
    if pox_name == a4_name:
        print("✓ FIX WORKING — Agent 4 explaining pox disease")
    else:
        print("✗ STILL BROKEN — mismatch")
else:
    print("\nPox not triggered on this image.")
    print("Try an image that triggers pox, or just check console output above.")