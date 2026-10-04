
#ifdef IRON_AUDIO

#include <alsa/asoundlib.h>
#include <iron_audio.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

iron_a2_buffer_t a2_buffer;
pthread_t        threadid;
bool             audio_running = false;
snd_pcm_t       *playback_handle;

static unsigned int      samples_per_second = 44100;
static bool              initialized        = false;
static snd_pcm_uframes_t period_size        = 2048;
static short             buf[4096];

uint32_t iron_a2_samples_per_second(void) {
	return samples_per_second;
}

static void copy_sample(void *buffer) {
	float left_value  = a2_buffer.channels[0][a2_buffer.read_location];
	float right_value = a2_buffer.channels[1][a2_buffer.read_location];
	a2_buffer.read_location += 1;
	if (a2_buffer.read_location >= a2_buffer.data_size) {
		a2_buffer.read_location = 0;
	}
	((int16_t *)buffer)[0] = (int16_t)(left_value * 32767);
	((int16_t *)buffer)[1] = (int16_t)(right_value * 32767);
}

void *audio_thread(void *arg) {
	if (snd_pcm_open(&playback_handle, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
		fprintf(stderr, "Error: cannot open audio device\n");
		return NULL;
	}

	int                  dir = 0;
	snd_pcm_hw_params_t *hw_params;
	snd_pcm_hw_params_malloc(&hw_params);
	snd_pcm_hw_params_any(playback_handle, hw_params);
	snd_pcm_hw_params_set_access(playback_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
	snd_pcm_hw_params_set_format(playback_handle, hw_params, SND_PCM_FORMAT_S16_LE);
	snd_pcm_hw_params_set_rate_near(playback_handle, hw_params, &samples_per_second, &dir);
	snd_pcm_hw_params_set_channels(playback_handle, hw_params, 2);
	snd_pcm_hw_params_set_period_size_near(playback_handle, hw_params, &period_size, &dir);
	snd_pcm_uframes_t buffer_size = period_size * 4;
	snd_pcm_hw_params_set_buffer_size_near(playback_handle, hw_params, &buffer_size);
	snd_pcm_hw_params(playback_handle, hw_params);
	snd_pcm_hw_params_get_period_size(hw_params, &period_size, &dir);
	snd_pcm_hw_params_free(hw_params);
	snd_pcm_prepare(playback_handle);

	while (audio_running) {
		if (iron_a2_internal_callback(&a2_buffer, (int)period_size)) {
			for (int i = 0; i < (int)period_size; i++) {
				copy_sample(&buf[i * 2]);
			}
		}
		else {
			memset(buf, 0, period_size * 2 * sizeof(short));
		}

		snd_pcm_sframes_t written;
		while ((written = snd_pcm_writei(playback_handle, buf, period_size)) < 0) {
			written = snd_pcm_recover(playback_handle, (int)written, 0);
			if (written < 0) {
				goto done;
			}
		}
	}

done:
	snd_pcm_close(playback_handle);
	return NULL;
}

void iron_a2_init() {
	if (initialized) {
		return;
	}

	iron_a2_internal_init();
	initialized = true;

	a2_buffer.read_location  = 0;
	a2_buffer.write_location = 0;
	a2_buffer.data_size      = 128 * 1024;
	a2_buffer.channel_count  = 2;
	a2_buffer.channels[0]    = (float *)malloc(a2_buffer.data_size * sizeof(float));
	a2_buffer.channels[1]    = (float *)malloc(a2_buffer.data_size * sizeof(float));

	audio_running = true;
	pthread_create(&threadid, NULL, &audio_thread, NULL);
}

void iron_a2_shutdown() {
	audio_running = false;
}

#endif
