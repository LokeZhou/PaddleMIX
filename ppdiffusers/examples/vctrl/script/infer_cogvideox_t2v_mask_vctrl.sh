
python infer_cogvideox_t2v_vctrl_cli.py \
  --pretrained_model_name_or_path "paddlemix/cogvideox-5b-vctrl" \
  --vctrl_path "vctrl_5b_t2v_mask.pdparams" \
  --vctrl_config "vctrl_configs/cogvideox_5b_vctrl_config.json" \
  --control_video_path "guide_values_1.mp4" \
  --control_mask_video_path 'mask_values_1.mp4' \
  --output_dir "infer_outputs/mask2video" \
  --prompt "" \
  --task "mask" \
  --width 720 \
  --height 480 \
  --max_frame 49 \
  --guidance_scale 3.5 \
  --num_inference_steps 25
