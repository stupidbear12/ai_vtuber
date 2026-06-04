import asyncio, sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'step3_vtube'))
import pyvts

async def main():
    plugin_info = {
        'plugin_name': 'emeth-controller',
        'developer': 'emeth',
        'authentication_token_path': './step3_vtube/'
    }
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()
    print('Authenticated!')

    t0 = time.time()
    print('Starting idle animation for 20 seconds...')
    while time.time() - t0 < 20:
        t = time.time() - t0
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4)
        head_x = 15 * math.sin(2 * math.pi * t / 5)
        head_y = 8 * math.sin(2 * math.pi * t / 7)
        head_z = 5 * math.sin(2 * math.pi * t / 9)
        blink_val = 1.0
        if int(t * 10) % 40 < 3:
            blink_val = 0.0

        send_data = {
            'parameterValues': [
                {'id': 'ParamBreath', 'value': breath},
                {'id': 'ParamAngleX', 'value': head_x},
                {'id': 'ParamAngleY', 'value': head_y},
                {'id': 'ParamAngleZ', 'value': head_z},
                {'id': 'ParamEyeLOpen', 'value': blink_val},
                {'id': 'ParamEyeROpen', 'value': blink_val},
                {'id': 'ParamMouthOpenY', 'value': 0.0},
            ]
        }
        request = vts.vts_request.BaseRequest('InjectParameterDataRequest', send_data)
        try:
            await vts.request(request)
        except Exception as e:
            print(f'Error: {e}')
            break
        await asyncio.sleep(0.033)

    print('Done!')
    await vts.close()

asyncio.run(main())
