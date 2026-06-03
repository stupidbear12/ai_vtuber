import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'step3_vtube'))
sys.path.insert(0, os.path.dirname(__file__))
from vtube_controller import VTubeController

async def main():
    ctrl = VTubeController()
    await ctrl.connect()
    try:
        await ctrl.start_full_animation()
        print('Full idle animation started! Running for 30 seconds...')
        await asyncio.sleep(30)
        print('Done')
    finally:
        await ctrl.stop_full_animation()
        await ctrl.disconnect()

asyncio.run(main())
