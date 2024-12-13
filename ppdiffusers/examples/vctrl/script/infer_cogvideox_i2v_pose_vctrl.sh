
python infer_cogvideox_5b_i2v_vctrl_cli.py \
  --pretrained_model_name_or_path "paddlemix/cogvideox-5b-i2v-vctrl" \
  --vctrl_path "vctrl_pose_5b_i2v.pdparams" \
  --vctrl_config "vctrl_configs/cogvideox_5b_i2v_vctrl_config.json" \
  --control_video_path "guide_values_1.mp4" \
  --ref_image_path "reference_image_1.jpg" \
  --control_mask_video_path 'mask_values_1.mp4' \
  --output_dir "infer_outputs/pose2video" \
  --prompt "" \
  --task "pose" \
  --width 480 \
  --height 720 \
  --max_frame 49 \
  --guidance_scale 3.5 \
  --num_inference_steps 25
