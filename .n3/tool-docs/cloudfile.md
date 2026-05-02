# cloudfile

Manages files on remote cloud storage using rclone.

IMPORTANT: This tool is restricted to only access the configured cloud storage path for security.

Usage notes:
- Use this tool to upload/download/delete/list files in the cloud storage
- All remote paths MUST start with the configured cloud storage prefix
- The remote must be pre-configured with `rclone config` on the system before using this tool

Parameters:
- operation (required): "upload", "download", "delete", or "list"
- source (required for upload/download): Source file path
  - For upload: local absolute path (e.g., "/path/to/video.mp4")
  - For download: MUST start with cloud storage prefix (e.g., "mydrive:folder/file.mp4")
- destination (required for upload/download): Destination path
  - For upload: MUST start with cloud storage prefix (e.g., "mydrive:folder/video.mp4")
  - For download: local absolute path (e.g., "/tmp/downloaded.mp4")
- remote_path (required for delete/list): Remote path MUST start with cloud storage prefix
  - For delete: Full path to file (e.g., "mydrive:folder/old_video.mp4")
  - For list: Directory path (e.g., "mydrive:folder" or "mydrive:folder/subfolder")
- transfer_mode (optional, upload/download only): "copy" (default) or "move" (deletes source after)
- show_progress (optional, upload/download only): Display transfer progress (default: true)

Examples:
- Upload video to Google Drive:
  operation="upload", source="/path/to/final_video.mp4", destination="mydrive:folder/video.mp4"

- Download file from Google Drive:
  operation="download", source="mydrive:folder/report.pdf", destination="/local/path/report.pdf"

- Move file to cloud (deletes local after upload):
  operation="upload", source="/tmp/log.txt", destination="mydrive:folder/log.txt", transfer_mode="move"

- Delete a remote file:
  operation="delete", remote_path="mydrive:folder/old_video.mp4"

- List files in remote folder:
  operation="list", remote_path="mydrive:folder"

Notes:
- File transfers are verified after completion by checking file sizes
- Large files may take time to transfer; the tool has a 10-minute timeout
- Ensure rclone remote is properly configured before use (run `rclone config` to set up)