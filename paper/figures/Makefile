ALL_FILES=abctoy.pdf

default: $(ALL_FILES)

%.pdf: %.py
	python -W ignore $<

clean:
	rm -f $(ALL_FILES)
