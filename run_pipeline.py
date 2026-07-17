from adcg.config import parse_config
from adcg.pipeline import run_pipeline


def main():
    config = parse_config()

    result = run_pipeline(
        image_path=config.image_path,
        info_path=config.info_path,
        output_dir=config.output_dir,
        gpt_model=config.gpt_model,
        direction=config.direction,
        layout_mode=config.layout_mode,
        seed=config.seed,
        cpu_offload=config.cpu_offload,
    )

    print("\n[PIPELINE DONE]")
    print(f"Final image: {result.final_image}")


if __name__ == "__main__":
    main()