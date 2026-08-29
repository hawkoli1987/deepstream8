/*
 * nvof_extract.c — dump per-frame NVIDIA Optical Flow Accelerator motion-vector
 * grids from a single .mp4 (h264) file to a flat binary + an index CSV.
 *
 * Element chain matches the verified-good T4 pipeline exactly:
 *   filesrc ! qtdemux ! h264parse ! nvv4l2decoder
 *     ! nvstreammux(batch-size=1, WxH) ! nvof(preset SLOW) ! fakesink
 * A buffer probe on nvof's src pad reads NvDsOpticalFlowMeta and writes:
 *   argv[2]  raw little-endian int16 pairs (flowx,flowy) per block, row-major,
 *            frames concatenated in decode order
 *   argv[3]  CSV  frame_num,rows,cols,mv_size,byte_offset,byte_len
 *
 * Usage: nvof_extract <file.mp4> <out.raw> <out_index.csv> <width> <height>
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"
#include "nvds_opticalflow_meta.h"

static FILE *g_raw = NULL;
static FILE *g_idx = NULL;
static long  g_offset = 0;
static int   g_frames = 0;
static GstElement *g_streammux = NULL;

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("EOS\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING: {
      gchar *dbg = NULL; GError *err = NULL;
      gst_message_parse_warning (msg, &err, &dbg);
      g_printerr ("WARN %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
      g_free (dbg); g_error_free (err);
      break;
    }
    case GST_MESSAGE_ERROR: {
      gchar *dbg = NULL; GError *err = NULL;
      gst_message_parse_error (msg, &err, &dbg);
      g_printerr ("ERROR %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
      if (dbg) g_printerr ("  %s\n", dbg);
      g_free (dbg); g_error_free (err);
      g_main_loop_quit (loop);
      break;
    }
    default: break;
  }
  return TRUE;
}

static GstPadProbeReturn
nvof_src_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta)
    return GST_PAD_PROBE_OK;

  for (NvDsMetaList * l = batch_meta->frame_meta_list; l != NULL; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *) l->data;
    for (NvDsMetaList * lu = fm->frame_user_meta_list; lu != NULL; lu = lu->next) {
      NvDsUserMeta *um = (NvDsUserMeta *) lu->data;
      if (um->base_meta.meta_type != NVDS_OPTICAL_FLOW_META)
        continue;

      NvDsOpticalFlowMeta *of = (NvDsOpticalFlowMeta *) um->user_meta_data;
      guint rows = of->rows, cols = of->cols;
      size_t nbytes = (size_t) rows * cols * sizeof (NvOFFlowVector);

      size_t w = fwrite (of->data, 1, nbytes, g_raw);
      if (w != nbytes)
        g_printerr ("short write frame %lu (%zu/%zu)\n",
            (unsigned long) of->frame_num, w, nbytes);

      fprintf (g_idx, "%lu,%u,%u,%u,%ld,%zu\n",
          (unsigned long) of->frame_num, rows, cols, of->mv_size,
          g_offset, nbytes);
      fflush (g_idx);

      g_offset += (long) nbytes;
      g_frames++;
      g_printerr ("frame %lu rows=%u cols=%u mv_size=%u bytes=%zu\n",
          (unsigned long) of->frame_num, rows, cols, of->mv_size, nbytes);
    }
  }
  return GST_PAD_PROBE_OK;
}

/* qtdemux has a sometimes-pad for video; link it to h264parse when it shows up */
static void
on_demux_pad (GstElement * demux, GstPad * pad, gpointer h264parse)
{
  GstCaps *caps = gst_pad_get_current_caps (pad);
  const gchar *name = gst_structure_get_name (gst_caps_get_structure (caps, 0));
  if (g_str_has_prefix (name, "video/x-h264")) {
    GstPad *sink = gst_element_get_static_pad (GST_ELEMENT (h264parse), "sink");
    if (gst_pad_link (pad, sink) != GST_PAD_LINK_OK)
      g_printerr ("qtdemux -> h264parse link failed\n");
    gst_object_unref (sink);
  }
  gst_caps_unref (caps);
}

int
main (int argc, char *argv[])
{
  if (argc != 6) {
    g_printerr ("Usage: %s <file.mp4> <out.raw> <out_index.csv> <width> <height>\n",
        argv[0]);
    return 1;
  }
  const gchar *path = argv[1];
  int W = atoi (argv[4]);
  int H = atoi (argv[5]);

  g_raw = fopen (argv[2], "wb");
  g_idx = fopen (argv[3], "w");
  if (!g_raw || !g_idx) { perror ("fopen"); return 1; }
  fprintf (g_idx, "frame_num,rows,cols,mv_size,byte_offset,byte_len\n");

  gst_init (&argc, &argv);
  GMainLoop *loop = g_main_loop_new (NULL, FALSE);
  GstElement *pipeline = gst_pipeline_new ("nvof-extract");

  GstElement *src     = gst_element_factory_make ("filesrc", NULL);
  GstElement *demux   = gst_element_factory_make ("qtdemux", NULL);
  GstElement *parse   = gst_element_factory_make ("h264parse", NULL);
  GstElement *dec     = gst_element_factory_make ("nvv4l2decoder", NULL);
  g_streammux         = gst_element_factory_make ("nvstreammux", "mux");
  GstElement *nvof    = gst_element_factory_make ("nvof", "nvof");
  GstElement *sink    = gst_element_factory_make ("fakesink", NULL);

  if (!pipeline || !src || !demux || !parse || !dec || !g_streammux || !nvof || !sink) {
    g_printerr ("element creation failed\n");
    return 1;
  }

  g_object_set (src, "location", path, NULL);
  g_object_set (g_streammux, "batch-size", 1, "width", W, "height", H,
      "live-source", 0, "batched-push-timeout", 4000000, NULL);
  g_object_set (nvof, "preset-level", 2, NULL);   /* SLOW = best quality */
  g_object_set (sink, "sync", FALSE, "async", FALSE, NULL);

  gst_bin_add_many (GST_BIN (pipeline), src, demux, parse, dec, g_streammux,
      nvof, sink, NULL);

  if (!gst_element_link (src, demux)) {
    g_printerr ("filesrc -> qtdemux link failed\n"); return 1;
  }
  if (!gst_element_link (parse, dec)) {
    g_printerr ("h264parse -> nvv4l2decoder link failed\n"); return 1;
  }
  {
    GstPad *dsrc = gst_element_get_static_pad (dec, "src");
    GstPad *msink = gst_element_request_pad_simple (g_streammux, "sink_0");
    if (!dsrc || !msink || gst_pad_link (dsrc, msink) != GST_PAD_LINK_OK) {
      g_printerr ("nvv4l2decoder -> streammux link failed\n"); return 1;
    }
    if (dsrc) gst_object_unref (dsrc);
    if (msink) gst_object_unref (msink);
  }
  if (!gst_element_link_many (g_streammux, nvof, sink, NULL)) {
    g_printerr ("streammux -> nvof -> sink link failed\n"); return 1;
  }
  g_signal_connect (demux, "pad-added", G_CALLBACK (on_demux_pad), parse);

  GstPad *ofsrc = gst_element_get_static_pad (nvof, "src");
  gst_pad_add_probe (ofsrc, GST_PAD_PROBE_TYPE_BUFFER, nvof_src_probe, NULL, NULL);
  gst_object_unref (ofsrc);

  GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  guint bus_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_main_loop_run (loop);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  g_source_remove (bus_id);
  g_main_loop_unref (loop);

  fclose (g_raw);
  fclose (g_idx);
  g_printerr ("done: %d frames of MV data\n", g_frames);
  return g_frames > 0 ? 0 : 2;
}
