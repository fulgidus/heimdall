#!/usr/bin/env python3
"""
Simula la logica del check_websdrs_health per capire se ritorna offline.
"""

# Simula il risultato dal task (da quello che vediamo nei log)
task_result = {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: False}

# Simula i WebSDR config
websdrs_config = [
    {"id": 1, "name": "SDR1"},
    {"id": 2, "name": "SDR2"},
    {"id": 3, "name": "SDR3"},
    {"id": 4, "name": "SDR4"},
    {"id": 5, "name": "SDR5"},
    {"id": 6, "name": "SDR6"},
    {"id": 7, "name": "SDR7"},
]

# Simula la logica dell'endpoint
health_status = {}
for ws_config in websdrs_config:
    ws_id = ws_config['id']
    is_online = task_result.get(ws_id, False)  # ← Questo è il test
    
    health_status[ws_id] = {
        'websdr_id': ws_id,
        'name': ws_config['name'],
        'status': 'online' if is_online else 'offline',
    }
    
    if not is_online:
        health_status[ws_id]['error_message'] = 'Health check failed or timed out'

print("=" * 80)
print("SIMULATED HEALTH CHECK RESPONSE")
print("=" * 80)
print(f"Task result: {task_result}")
print()

for ws_id, status in health_status.items():
    print(f"SDR #{ws_id}: {status['status']}")

print()
print("=" * 80)

# Verifica 
all_offline = all(v['status'] == 'offline' for v in health_status.values())
if all_offline:
    print("❌ BUG FOUND: All offline!")
    print("\nDEBUG:")
    for ws_config in websdrs_config:
        ws_id = ws_config['id']
        result_value = task_result.get(ws_id, False)
        print(f"  ws_id={ws_id} (type={type(ws_id).__name__}), task_result.get({ws_id})={result_value}")
else:
    print("✅ Logic is correct - should show 6 online, 1 offline")
