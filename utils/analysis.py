import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging, random

import numpy as np
import tensorflow as tf

from collections import defaultdict, OrderedDict

def calc_sims(latents):
    dot_mat = np.dot(latents, latents.T)
    nrm_mat = np.linalg.norm(latents, axis=1)
    mlt_mat = np.outer(nrm_mat, nrm_mat)
    sims = dot_mat / mlt_mat
    return sims

def get_closest(centroid, latents):
    dists = (latents - centroid)**2
    dists = np.sum(dists, axis=1)
    dists = np.sqrt(dists)

    dists = list(enumerate(dists))
    dists = sorted(dists, key=lambda el: el[1])
    latent_idcs, dists = zip(*dists)
    return latent_idcs, dists

def calc_metrics(latents, labels, sims, num_labels, prec_ranks):
    mean_latents = np.zeros([num_labels, latents.shape[1]])
    rel_sim_by_label = np.zeros(num_labels)
    oth_sim_by_label = np.zeros(num_labels)
    label_precision = defaultdict(lambda: np.zeros(num_labels))
    label_idcs = [[] for _ in range(num_labels)]

    for idx in range(latents.shape[0]):
        sys.stdout.write("\rCalculating metrics for %d/%d (%.2f%%)..." % (idx+1, latents.shape[0], ((idx+1)*100)/latents.shape[0]))
        sys.stdout.flush()

        lbl = labels[idx]
        # sort neighbours by similarity
        cur_sims = list(enumerate(sims[idx]))
        cur_sims = sorted(cur_sims, key=lambda el: el[1], reverse=True)
        # get sorted neighbours and similarities
        sim_idcs, sim_vals = zip(*cur_sims)
        sim_idcs, sim_vals = np.array(sim_idcs), np.array(sim_vals)

        # calculate average distances
        rel_idcs = np.where(labels == (lbl * np.ones_like(labels)))
        oth_idcs = np.where(labels != (lbl * np.ones_like(labels)))
        rel_avg_sim = np.mean(sims[idx][rel_idcs])
        oth_avg_sim = np.mean(sims[idx][oth_idcs])
        rel_sim_by_label[lbl] += rel_avg_sim
        oth_sim_by_label[lbl] += oth_avg_sim
        label_idcs[lbl].append(idx)

        # calculate precision/recall at top n
        for rank in prec_ranks:
            # get top n
            top_idcs, top_vals = sim_idcs[:rank+1], sim_vals[:rank+1]
            # count TP/FP and calculate precision
            tp = np.sum(labels[top_idcs] == lbl)
            fp = np.sum(labels[top_idcs] != lbl)
            precision = tp / (tp + fp)
            # store results
            label_precision[rank][lbl] += precision

    # compute mean latents
    for lbl in range(num_labels):
        mean_latents[lbl] = np.mean(latents[label_idcs[lbl]], axis=0)

    # average out metrics
    label_count = np.array([len(lbl_idcs) for lbl_idcs in label_idcs])
    rel_sim_by_label /= label_count
    oth_sim_by_label /= label_count
    for rank in prec_ranks:
        label_precision[rank] /= label_count

    logging.info("\rCalculated metrics for %d latents.%s" % (latents.shape[0], ' '*16))

    return mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision

def log_metrics(label_descs, top_n, rel_sim_by_label, oth_sim_by_label, precision_by_label):
    logging.info("Overall metrics:")
    for label_idx, label in enumerate(label_descs):
        logging.info("  %s: %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            label, precision_by_label[label_idx], top_n,
            rel_sim_by_label[label_idx],
            oth_sim_by_label[label_idx]
            ))
    logging.info("Total (avg): %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            np.mean(precision_by_label), top_n,
            np.mean(rel_sim_by_label),
            np.mean(oth_sim_by_label)
        ))

def get_sorted_triplets(latents):
    # set up non-overlapping trios
    trio_keys = [
        tuple(sorted([i1, i2, i3]))
        for i1 in range(latents.shape[0])
            for i2 in range(latents.shape[0])
                for i3 in range(latents.shape[0])
                    if len(set([i1, i2, i3])) > 2
    ]
    # calculate trio similarities
    trio_sims = {}
    for trio_key in trio_keys:
        trio_sims[trio_key] = np.linalg.norm(latents[[trio_key[0]]] - latents[[trio_key[1]]])\
            + np.linalg.norm(latents[[trio_key[1]]] - latents[[trio_key[2]]])\
            + np.linalg.norm(latents[[trio_key[2]]] - latents[[trio_key[0]]])

    sorted_triplets = sorted(list(trio_sims.items()), key=lambda el: el[1], reverse=True)
    trio_keys, trio_dists = zip(*sorted_triplets)
    return trio_keys, trio_dists


def gen_eval_task(mean_latents, latents, num_examples, num_tasks):
    # get triplet of means with largest distance between them
    trio_keys, trio_dists = get_sorted_triplets(mean_latents)
    eval_trio, eval_trio_dist = trio_keys[0], trio_dists[0]
    print("Calculated mean triplet %s with cumulative Euclidean distance %.2f." % (str(eval_trio), eval_trio_dist))
    # get samples which lie closest to respective means
    trio_sample_idcs = np.zeros([3, num_examples + num_tasks], dtype=int)
    for tidx in range(3):
        closest_idcs, closest_dists = get_closest(mean_latents[eval_trio[tidx]], latents)
        trio_sample_idcs[tidx] = closest_idcs[:num_examples + num_tasks]
        avg_dist = np.mean(closest_dists[:num_examples + num_tasks])
        print("Calculated %d samples for mean %d with average distance %.2f." % (trio_sample_idcs[tidx].shape[0], eval_trio[tidx], avg_dist))
    # get examples
    example_idcs = sorted(np.random.choice((num_examples + num_tasks), num_examples, replace=False))
    examples = np.squeeze(trio_sample_idcs[:,example_idcs].flatten())
    examples = examples.tolist()
    trio_sample_idcs[:,example_idcs] = -1
    # get tasks
    task_idcs = np.where(trio_sample_idcs >= 0.)
    task_trios = np.reshape(trio_sample_idcs[task_idcs], [3, num_tasks])
    task_trios = [task_trios[:, i].tolist() for i in range(num_tasks)]
    # randomly select truths for tasks
    tasks = []
    for trio in task_trios:
        truth_idx = random.randint(0, 3)
        tasks.append(OrderedDict([
            ('truth', truth_idx),
            ('other', trio[:truth_idx] + trio[truth_idx + 1:])
        ]))
    return examples, tasks
