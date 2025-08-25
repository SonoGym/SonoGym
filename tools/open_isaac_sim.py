import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument('filename')
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
import isaaclab.sim as sim_utils

def main():
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, gravity=[0.0, 0.0, 0.0])
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg(size=(500.0,500.0))
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    # lights
    cfg_light_distant = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant)


    cfg = add_reference_to_stage(usd_path=args_cli.filename, prim_path="/Robot")

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


# Close the simulator
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
