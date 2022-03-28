for rcp in rcp26 rcp45 rcp60 rcp85
do
	for man in base tobd tobe toCoppice tone unmanaged
	do

		simdir=${man}_${rcp}
		mkdir -p ${simdir}
		for file in cpool.out seasonality.out fpc.out cpool_forest.out converted_fraction.out dens.out cmass_harv_per_species_stem.out cmass_harv_per_species_slow.out cflux.out diamstruct_forest.out diversity.out canopy_height.out active_fraction_forest.out fpc_forest.out cmass_luc_per_species_stem.out cmass_luc_per_species_slow.out cmass_harv_per_species_residue_to_atm.out cmass_luc_per_species_residue_to_atm.out mpsi_s_upper_forest.out mpsi_s_lower_forest.out maet_forest.out mintercep_forest.out mevap_forest.out cflux_forest.out crownarea_forest.out
		do
			head -1 /home/konni/Documents/konni/projekte/phd/runs/cluster_runs/dist_no_fire_in_managed/${simdir}/$file > ${simdir}/$file


			for gc in " 9.75 +49.75" "13.75 +55.75" "23.75 +61.75" "21.75 +37.75"
			do
				grep -E "${gc} +(19[8-9].*|20.*|21[0-2].*|2130)" /home/konni/Documents/konni/projekte/phd/runs/cluster_runs/dist_no_fire_in_managed/${simdir}/$file >> ${simdir}/$file
			done
		done
	done
done

