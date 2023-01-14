
# def hemefinder(
#     target: str,
#     outputdir: str,
#     coordinators: int or list
# ):
#     start = time.time()

#     # Load stats for bimodal distributions
#     stats = load_stats()
#     stats_res = load_stats_res()

#     # Read protein and find possible coordinating residues
#     atomic = read_pdb(target, outputdir)
#     alphas, betas, res_number_coordinators, all_alphas, all_betas, residues_names = parse_residues(atomic, coordinators, stats)

#     # Detect cavities and analyse possible coordinations
#     dic_coordinating = {}
#     probes = volume_pyKVFinder(atomic, target, outputdir)
#     scores = coordination_score(alphas, betas, stats, probes, res_number_coordinators)
#     coord_residues = clustering(scores, dic_coordinating)
#     coord_residues_centroid = centroid(coord_residues)
    
#     coord_residues_new = new_probes(coord_residues_centroid, alphas, betas, stats, res_number_coordinators)
#     coord_residues_centroid_new = centroid(coord_residues_new)
#     sorted_results = {k:v for k,v in sorted(coord_residues_centroid_new.items(), key=lambda x: x[1]['score'], reverse=True)}
    
#     final_results={}

#     for k, v in sorted_results.items():
#         sphere = detect_ellipsoid(probes, v['centroid'])
#         yes_no = elipsoid(sphere)
#         if yes_no == True:
#             final_results[k] = v
#             final_results[k]['elipsoid'] = np.array(sphere)
#             residues = detect_residues(sphere, all_alphas, all_betas, residues_names)
#             score = residue_scoring(residues,stats_res)
#             final_results[k]['score_res'] = score

#     basename = Path(target).stem
#     outputfile = os.path.join(outputdir, basename)

#     for k,v in final_results.items():
#         print(k, v['score'])

#     print(outputfile)
#     create_PDB(final_results, outputfile)


        
#     end = time.time()
#     print(f'\nComputation took {round(end - start, 2)} s')


# if __name__ == '__main__':
#     help(hemefinder)
